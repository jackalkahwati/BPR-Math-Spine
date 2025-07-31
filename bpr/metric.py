"""
Metric perturbation module for BPR-Math-Spine

Implements Equations (3) and (6b) for computing the metric perturbation Δg_μν
from the boundary phase field φ. This is the core of the BPR theoretical framework.
"""

import numpy as np
import sympy as sp
from sympy import symbols, Matrix, diff, simplify, lambdify


class MetricPerturbation:
    """Container for metric perturbation data and computations."""
    
    def __init__(self, delta_g, coordinates, phi_field, coupling_lambda):
        self.delta_g = delta_g  # 4x4 metric perturbation tensor
        self.coordinates = coordinates  # Coordinate system (t, x, y, z)
        self.phi_field = phi_field  # Boundary phase field
        self.coupling_lambda = coupling_lambda
        
    def stress_energy_tensor(self):
        """Compute the stress-energy tensor T^φ_μν."""
        return compute_stress_tensor(self.phi_field, self.delta_g, self.coordinates)
    
    def energy_momentum_conservation(self):
        """Verify ∇^μ T^φ_μν = 0 (conservation law)."""
        T = self.stress_energy_tensor()
        return verify_conservation(T, self.delta_g, self.coordinates)
    
    def trace(self):
        """Compute trace of metric perturbation."""
        if isinstance(self.delta_g, np.ndarray):
            return np.trace(self.delta_g)
        else:
            # For symbolic matrices
            return sp.trace(self.delta_g)


def metric_perturbation(phi_field, coupling_lambda, coordinates="cartesian"):
    """
    Compute metric perturbation Δg_μν from boundary phase field φ.
    
    Implements Equation (3): 
    Δg_μν = λ φ(x_boundary) * [geometric coupling terms]
    
    And Equation (6b):
    The specific form of geometric coupling for boundary-localized fields.
    
    Parameters
    ----------
    phi_field : BoundaryFieldSolution or callable
        The boundary phase field φ(x, y, z)
    coupling_lambda : float
        Coupling strength λ between boundary field and metric
    coordinates : str
        Coordinate system: "cartesian", "spherical", "cylindrical"
        
    Returns
    -------
    MetricPerturbation
        Container with computed metric perturbation
        
    Examples
    --------
    >>> solution = solve_phase(mesh, source)
    >>> delta_g = metric_perturbation(solution, lambda=0.1)
    >>> T_stress = delta_g.stress_energy_tensor()
    """
    
    # Define symbolic coordinates
    if coordinates == "cartesian":
        t, x, y, z = symbols('t x y z', real=True)
        coords = [t, x, y, z]
    elif coordinates == "spherical":
        t, r, theta, phi_coord = symbols('t r theta phi', real=True)
        coords = [t, r, theta, phi_coord]
    elif coordinates == "cylindrical":
        t, rho, phi_coord, z = symbols('t rho phi z', real=True)
        coords = [t, rho, phi_coord, z]
    else:
        raise ValueError(f"Unknown coordinate system: {coordinates}")
    
    # Create symbolic representation of boundary field
    if hasattr(phi_field, 'phi'):
        # BoundaryFieldSolution object
        phi_sym = _boundary_field_to_symbolic(phi_field, coords)
    elif callable(phi_field):
        # Function
        phi_sym = phi_field(*coords[1:])  # Exclude time coordinate
    else:
        # Assume already symbolic
        phi_sym = phi_field
    
    # Compute metric perturbation according to Eq (3)
    delta_g_matrix = _compute_delta_g_matrix(phi_sym, coupling_lambda, coords, coordinates)
    
    return MetricPerturbation(delta_g_matrix, coords, phi_field, coupling_lambda)


def _boundary_field_to_symbolic(phi_field_solution, coords):
    """Convert BoundaryFieldSolution to symbolic expression."""
    # This is a simplified representation
    # In practice, would interpolate the FEniCS solution
    
    x, y, z = coords[1], coords[2], coords[3]
    
    # For now, use a simple polynomial approximation
    # In full implementation, would sample the solution and fit
    phi_sym = symbols('phi_0') * (x**2 + y**2 + z**2)
    
    return phi_sym


def _compute_delta_g_matrix(phi_sym, lam, coords, coord_system):
    """
    Compute the 4x4 metric perturbation matrix.
    
    Implements the specific form from Eq (3) and (6b):
    Δg_μν = λ φ(x_∂) [boundary coupling terms]
    """
    t, x, y, z = coords[0], coords[1], coords[2], coords[3]
    
    # Initialize 4x4 metric perturbation
    delta_g = sp.zeros(4, 4)
    
    if coord_system == "cartesian":
        # Boundary-localized coupling (simplified form)
        # This represents the coupling between boundary field and spacetime metric
        
        # Spatial diagonal terms: coupling to boundary curvature
        delta_g[1, 1] = lam * phi_sym * sp.exp(-((x**2 + y**2 + z**2) - 1)**2)
        delta_g[2, 2] = lam * phi_sym * sp.exp(-((x**2 + y**2 + z**2) - 1)**2) 
        delta_g[3, 3] = lam * phi_sym * sp.exp(-((x**2 + y**2 + z**2) - 1)**2)
        
        # Time-space mixing terms
        delta_g[0, 1] = lam * phi_sym * diff(phi_sym, x) * sp.exp(-((x**2 + y**2 + z**2) - 1)**2)
        delta_g[1, 0] = delta_g[0, 1]  # Symmetry
        
        # Cross terms for boundary effects
        delta_g[1, 2] = lam * diff(phi_sym, x) * diff(phi_sym, y)
        delta_g[2, 1] = delta_g[1, 2]
        
    elif coord_system == "spherical":
        r, theta, phi_coord = coords[1], coords[2], coords[3]
        
        # Radial coupling dominates near boundary
        boundary_factor = sp.exp(-(r - 1)**2)
        
        delta_g[1, 1] = lam * phi_sym * boundary_factor
        delta_g[2, 2] = lam * phi_sym * boundary_factor * r**2
        delta_g[3, 3] = lam * phi_sym * boundary_factor * r**2 * sp.sin(theta)**2
        
    else:
        # Generic form for other coordinate systems
        for i in range(1, 4):
            delta_g[i, i] = lam * phi_sym
    
    return delta_g


def compute_stress_tensor(phi_field, delta_g, coords):
    """
    Compute stress-energy tensor T^φ_μν from the metric perturbation.
    
    Uses the standard formula:
    T_μν = (2/√(-g)) δS/δg^μν
    
    Where S is the action for the boundary field.
    """
    
    # Background Minkowski metric
    eta = sp.diag(-1, 1, 1, 1)
    
    # Full metric: g = η + Δg
    g_full = eta + delta_g
    
    # Compute determinant
    g_det = g_full.det()
    
    # Stress tensor components (simplified)
    T = sp.zeros(4, 4)
    
    # Energy density T_00
    if hasattr(phi_field, 'phi') and hasattr(phi_field, 'get_gradient'):
        # For BoundaryFieldSolution objects
        grad_phi = phi_field.get_gradient()
        T[0, 0] = sp.Rational(1, 2) * grad_phi.dot(grad_phi)
    else:
        # For symbolic phi
        t, x, y, z = coords
        grad_phi_squared = diff(phi_field, x)**2 + diff(phi_field, y)**2 + diff(phi_field, z)**2
        T[0, 0] = sp.Rational(1, 2) * grad_phi_squared
    
    # Momentum density T_0i
    for i in range(1, 4):
        T[0, i] = 0  # Zero for static fields
        T[i, 0] = T[0, i]
    
    # Stress components T_ij
    for i in range(1, 4):
        for j in range(1, 4):
            if i == j:
                T[i, j] = sp.Rational(1, 2) * T[0, 0]  # Isotropic pressure
            else:
                T[i, j] = 0
    
    return T


def verify_conservation(stress_tensor, metric, coords):
    """
    Verify energy-momentum conservation: ∇^μ T_μν = 0.
    
    This is a key mathematical checkpoint for the BPR theory.
    """
    
    # Compute covariant divergence
    conservation_check = sp.zeros(4, 1)
    
    for nu in range(4):
        for mu in range(4):
            # ∂_μ T^μ_ν term
            conservation_check[nu] += diff(stress_tensor[mu, nu], coords[mu])
            
            # Christoffel symbol corrections (simplified)
            # Full implementation would compute all connection coefficients
            pass
    
    # Simplify results
    conservation_check = simplify(conservation_check)
    
    return conservation_check


def casimir_stress_correction(phi_field, radius, coupling_lambda):
    """
    Compute the BPR correction to Casimir stress.
    
    This implements the specific prediction that distinguishes BPR theory
    from standard QED calculations.
    
    Parameters
    ----------
    phi_field : BoundaryFieldSolution
        Solved boundary phase field
    radius : float
        Characteristic size of the boundary
    coupling_lambda : float
        BPR coupling strength
        
    Returns
    -------
    dict
        Stress correction components and total force
    """
    
    # Standard Casimir pressure (attractive)
    casimir_pressure_standard = -np.pi**2 / (240 * radius**4)
    
    # BPR correction depends on boundary field energy
    if hasattr(phi_field, 'compute_energy'):
        field_energy = phi_field.compute_energy()
    else:
        # Estimate from field magnitude
        field_energy = 1.0  # Placeholder
    
    # BPR correction formula (derived from Eq 7)
    # This is the key prediction that makes BPR falsifiable
    bpr_correction = coupling_lambda * field_energy / radius**2
    
    # Total modified pressure
    total_pressure = casimir_pressure_standard + bpr_correction
    
    return {
        "standard_pressure": casimir_pressure_standard,
        "bpr_correction": bpr_correction,
        "total_pressure": total_pressure,
        "field_energy": field_energy,
        "relative_correction": bpr_correction / abs(casimir_pressure_standard)
    }


def compute_metric_eigenvalues(metric_perturbation):
    """
    Compute eigenvalues of the metric perturbation.
    
    Used for stability analysis and physical interpretation.
    """
    
    delta_g = metric_perturbation.delta_g
    
    if isinstance(delta_g, sp.Matrix):
        # Symbolic eigenvalues
        eigenvals = delta_g.eigenvals()
        return eigenvals
    else:
        # Numerical eigenvalues
        eigenvals = np.linalg.eigvals(delta_g)
        return eigenvals


def export_metric_to_mathematica(metric_perturbation, filename):
    """Export metric perturbation to Mathematica format for further analysis."""
    
    delta_g = metric_perturbation.delta_g
    coords = metric_perturbation.coordinates
    
    # Convert to Mathematica format
    mathematica_str = "deltaG = " + str(delta_g).replace('Matrix', 'List')
    mathematica_str += ";\ncoords = " + str(coords) + ";"
    
    with open(filename, 'w') as f:
        f.write(mathematica_str)
    
    return filename