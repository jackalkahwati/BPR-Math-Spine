"""
Boundary field solver for BPR-Math-Spine

Solves the boundary Laplacian equation κ∇²_Σ φ = f (Eq 6a) using FEniCS.
This module implements the core PDE solver for the phase field on the boundary.
"""

import numpy as np
try:
    import fenics as fe
except ImportError:
    fe = None


class BoundaryFieldSolution:
    """Container for boundary field solution data."""
    
    def __init__(self, phi, mesh, function_space, boundary_mesh, source_term=None):
        self.phi = phi  # Solution function
        self.mesh = mesh
        self.function_space = function_space
        self.boundary_mesh = boundary_mesh
        self.source_term = source_term
        
    def evaluate_at_points(self, points):
        """Evaluate solution at given points."""
        if fe is None:
            raise ImportError("FEniCS required for evaluation")
        
        values = []
        for point in points:
            try:
                values.append(self.phi(point))
            except:
                values.append(np.nan)
        return np.array(values)
    
    def get_gradient(self):
        """Compute gradient of the solution."""
        if fe is None:
            raise ImportError("FEniCS required for gradient computation")
        
        return fe.grad(self.phi)
    
    def compute_energy(self):
        """Compute field energy ∫ |∇φ|² dS."""
        if fe is None:
            raise ImportError("FEniCS required for energy computation")
        
        grad_phi = fe.grad(self.phi)
        energy_density = fe.dot(grad_phi, grad_phi)
        return fe.assemble(energy_density * fe.dx(self.mesh))


def solve_phase(boundary_mesh, source, kappa=1.0, boundary_conditions=None):
    """
    Solve the boundary Laplacian equation κ∇²_Σ φ = f.
    
    This implements Equation (6a) from the BPR theory, solving for the 
    phase field φ on the boundary surface Σ.
    
    Parameters
    ----------
    boundary_mesh : BoundaryMesh
        The triangulated boundary surface
    source : callable or fe.Expression
        Source term f(x, y, z) in the equation
    kappa : float
        Diffusion coefficient κ
    boundary_conditions : dict, optional
        Boundary conditions (for boundaries of boundaries)
        
    Returns
    -------
    BoundaryFieldSolution
        Solution container with φ and metadata
        
    Examples
    --------
    >>> mesh = make_boundary(mesh_size=0.1)
    >>> source = lambda x, y, z: np.sin(x) * np.cos(y)
    >>> solution = solve_phase(mesh, source, kappa=1.5)
    >>> energy = solution.compute_energy()
    """
    if fe is None:
        raise ImportError("FEniCS not available. Install with: conda install fenics")
    
    mesh = boundary_mesh.mesh
    
    # Define function space (P1 elements on the surface)
    V = fe.FunctionSpace(mesh, "CG", 1)
    
    # Define trial and test functions
    phi = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    
    # Convert source to FEniCS expression if needed
    if callable(source):
        source_expr = _convert_callable_to_expression(source, mesh)
    else:
        source_expr = source
    
    # Define variational problem: κ ∇φ · ∇v dS = f v dS
    a = kappa * fe.dot(fe.grad(phi), fe.grad(v)) * fe.dx
    L = source_expr * v * fe.dx
    
    # Apply boundary conditions if specified
    bcs = []
    if boundary_conditions is not None:
        bcs = _apply_boundary_conditions(V, boundary_conditions)
    
    # Solve the linear system
    phi_solution = fe.Function(V)
    
    try:
        fe.solve(a == L, phi_solution, bcs)
    except Exception as e:
        # Try alternative solver if direct solver fails
        A, b = fe.assemble_system(a, L, bcs)
        solver = fe.KrylovSolver("cg", "ilu")
        solver.solve(A, phi_solution.vector(), b)
    
    return BoundaryFieldSolution(
        phi_solution, mesh, V, boundary_mesh, source_term=source_expr
    )


def solve_eigenvalue_problem(boundary_mesh, n_modes=10):
    """
    Solve the eigenvalue problem ∇²_Σ φ = λ φ for Laplacian eigenmodes.
    
    This is used for mathematical verification and benchmark calculations.
    
    Parameters
    ----------
    boundary_mesh : BoundaryMesh
        The boundary surface mesh
    n_modes : int
        Number of eigenmodes to compute
        
    Returns
    -------
    tuple
        (eigenvalues, eigenfunctions) arrays
    """
    if fe is None:
        raise ImportError("FEniCS required for eigenvalue computation")
    
    mesh = boundary_mesh.mesh
    V = fe.FunctionSpace(mesh, "CG", 1)
    
    # Define trial and test functions
    phi = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    
    # Assemble matrices for generalized eigenvalue problem
    # A φ = λ M φ where A is stiffness, M is mass
    a = fe.dot(fe.grad(phi), fe.grad(v)) * fe.dx
    m = phi * v * fe.dx
    
    A = fe.assemble(a)
    M = fe.assemble(m)
    
    # Convert to numpy for eigenvalue solving
    A_np = A.array()
    M_np = M.array()
    
    # Solve generalized eigenvalue problem
    try:
        from scipy.sparse.linalg import eigsh
        eigenvals, eigenvecs = eigsh(A_np, k=n_modes, M=M_np, sigma=0.1)
    except ImportError:
        # Fallback to numpy
        eigenvals, eigenvecs = np.linalg.eigh(
            np.linalg.solve(M_np, A_np)
        )
        eigenvals = eigenvals[:n_modes]
        eigenvecs = eigenvecs[:, :n_modes]
    
    # Convert eigenvectors back to FEniCS functions
    eigenfunctions = []
    for i in range(n_modes):
        phi_mode = fe.Function(V)
        phi_mode.vector()[:] = eigenvecs[:, i]
        eigenfunctions.append(phi_mode)
    
    return eigenvals, eigenfunctions


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
    """Apply boundary conditions to the function space."""
    bcs = []
    
    for bc_spec in boundary_conditions:
        if bc_spec["type"] == "dirichlet":
            value = bc_spec["value"]
            boundary = bc_spec["boundary"]
            
            if callable(value):
                # Convert to expression
                bc_expr = _convert_callable_to_expression(value, function_space.mesh())
            else:
                bc_expr = fe.Constant(value)
            
            bc = fe.DirichletBC(function_space, bc_expr, boundary)
            bcs.append(bc)
    
    return bcs


def verify_convergence(boundary_mesh_sizes, exact_solution=None):
    """
    Verify convergence of the boundary field solver.
    
    Parameters
    ----------
    boundary_mesh_sizes : list
        List of mesh sizes to test
    exact_solution : callable, optional
        Exact solution for comparison
        
    Returns
    -------
    dict
        Convergence analysis results
    """
    from .geometry import make_boundary
    
    errors = []
    
    for mesh_size in boundary_mesh_sizes:
        # Create mesh
        mesh = make_boundary(mesh_size=mesh_size)
        
        # Define test problem
        if exact_solution is None:
            # Use manufactured solution: φ = sin(π x) cos(π y)
            exact_phi = lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y)
            source = lambda x, y, z: 2 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
        else:
            exact_phi = exact_solution
            # Would need to compute Laplacian of exact_solution
            source = lambda x, y, z: 0  # Placeholder
        
        # Solve
        solution = solve_phase(mesh, source)
        
        # Compute error (simplified)
        try:
            # Sample solution at mesh vertices
            coordinates = mesh.mesh.coordinates()
            exact_values = np.array([
                exact_phi(coord[0], coord[1], coord[2] if len(coord) > 2 else 0)
                for coord in coordinates
            ])
            
            computed_values = solution.phi.vector().get_local()
            
            # L2 error
            error = np.sqrt(np.mean((exact_values - computed_values)**2))
            errors.append(error)
            
        except Exception:
            errors.append(np.nan)
    
    return {
        "mesh_sizes": boundary_mesh_sizes,
        "errors": errors,
        "convergence_rate": _estimate_convergence_rate(boundary_mesh_sizes, errors)
    }


def _estimate_convergence_rate(h_values, errors):
    """Estimate convergence rate from mesh size and error data."""
    valid_indices = ~np.isnan(errors)
    if np.sum(valid_indices) < 2:
        return np.nan
    
    h_clean = np.array(h_values)[valid_indices]
    e_clean = np.array(errors)[valid_indices]
    
    # Fit log(e) = log(C) + p * log(h)
    log_h = np.log(h_clean)
    log_e = np.log(e_clean)
    
    try:
        coeffs = np.polyfit(log_h, log_e, 1)
        return coeffs[0]  # Convergence rate p
    except:
        return np.nan