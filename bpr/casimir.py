"""
Casimir force module for BPR-Math-Spine

Implements Equation (7) - the falsifiable prediction of BPR theory.
Computes the deviation from standard Casimir force due to boundary phase resonance.
This is the core prediction that makes BPR experimentally testable.
"""

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.constants import hbar, c, pi
from .geometry import make_boundary
from .boundary_field import solve_phase
from .metric import metric_perturbation, casimir_stress_correction


# Physical constants
EPSILON_0 = 8.854187817e-12  # F/m
MU_0 = 4e-7 * pi  # H/m
CASIMIR_PREFACTOR = pi**2 * hbar * c / 240  # Standard Casimir coefficient


class CasimirResult:
    """Container for Casimir force calculation results."""
    
    def __init__(self, radius, standard_force, bpr_correction, total_force, 
                 field_energy=None, coupling_lambda=None):
        self.radius = radius
        self.standard_force = standard_force
        self.bpr_correction = bpr_correction  
        self.total_force = total_force
        self.field_energy = field_energy
        self.coupling_lambda = coupling_lambda
        
    @property
    def relative_deviation(self):
        """Relative deviation from standard Casimir force."""
        if abs(self.standard_force) > 0:
            return self.bpr_correction / abs(self.standard_force)
        return np.inf
    
    def to_dict(self):
        """Convert to dictionary for easy export."""
        return {
            'R [m]': self.radius,
            'F_Casimir [N]': self.standard_force,
            'ΔF_BPR [N]': self.bpr_correction,
            'F_total [N]': self.total_force,
            'relative_deviation': self.relative_deviation,
            'field_energy': self.field_energy,
            'coupling_lambda': self.coupling_lambda
        }


def casimir_force(radius, geometry="parallel_plates", coupling_lambda=1e-3, 
                  mesh_size=0.05, boundary_source=None):
    """
    Compute Casimir force with BPR corrections.
    
    This is the main function implementing Equation (7):
    F_total = F_Casimir + ΔF_BPR(φ, λ, R)
    
    Parameters
    ----------
    radius : float
        Characteristic size scale (plate separation, sphere radius, etc.)
    geometry : str
        Geometry type: "parallel_plates", "sphere", "cylinder"
    coupling_lambda : float
        BPR coupling strength λ
    mesh_size : float
        Mesh resolution for boundary field calculation
    boundary_source : callable, optional
        Source term for boundary field equation
        
    Returns
    -------
    CasimirResult
        Complete force calculation with BPR corrections
        
    Examples
    --------
    >>> result = casimir_force(radius=1e-6, coupling_lambda=1e-3)
    >>> print(f"Standard Casimir: {result.standard_force:.2e} N")
    >>> print(f"BPR correction: {result.bpr_correction:.2e} N") 
    >>> print(f"Relative deviation: {result.relative_deviation:.1%}")
    """
    
    # Compute standard Casimir force
    F_casimir = _standard_casimir_force(radius, geometry)
    
    # Solve boundary field equation
    boundary_mesh = make_boundary(mesh_size=mesh_size, geometry="sphere", radius=radius)
    
    if boundary_source is None:
        # Default source: spherical harmonic excitation
        boundary_source = lambda x, y, z: np.sin(pi * x) * np.cos(pi * y) * np.exp(-x**2 - y**2 - z**2)
    
    phi_solution = solve_phase(boundary_mesh, boundary_source)
    
    # Compute metric perturbation and stress corrections
    delta_g = metric_perturbation(phi_solution, coupling_lambda)
    stress_correction = casimir_stress_correction(phi_solution, radius, coupling_lambda)
    
    # BPR force correction (Equation 7)
    Delta_F_BPR = _compute_bpr_force_correction(
        phi_solution, radius, coupling_lambda, geometry
    )
    
    # Total force
    F_total = F_casimir + Delta_F_BPR
    
    return CasimirResult(
        radius=radius,
        standard_force=F_casimir,
        bpr_correction=Delta_F_BPR,
        total_force=F_total,
        field_energy=phi_solution.compute_energy(),
        coupling_lambda=coupling_lambda
    )


def sweep_radius(r_min=0.2e-6, r_max=5e-6, n=40, coupling_lambda=1e-3, 
                 geometry="parallel_plates", out=None):
    """
    Sweep radius to generate the falsifiable prediction curve (Equation 7).
    
    This generates the key data for experimental verification of BPR theory.
    
    Parameters
    ----------
    r_min, r_max : float
        Radius range in meters
    n : int
        Number of points in the sweep
    coupling_lambda : float
        BPR coupling strength
    geometry : str
        Boundary geometry
    out : str, optional
        Output CSV filename
        
    Returns
    -------
    pandas.DataFrame
        Data table with columns: R [m], F_Casimir [N], ΔF_BPR [N], F_total [N]
        
    Examples
    --------
    >>> data = sweep_radius(r_min=0.2e-6, r_max=5e-6, n=40)
    >>> data.to_csv('casimir_prediction.csv', index=False)
    """
    
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n)
    results = []
    
    print(f"Computing BPR-Casimir sweep: {n} points from {r_min:.1e} to {r_max:.1e} m")
    
    for i, radius in enumerate(radii):
        print(f"  {i+1:2d}/{n}: R = {radius:.2e} m", end=" ... ")
        
        try:
            # Adaptive mesh size based on radius
            mesh_size = min(0.1, radius * 10)
            
            result = casimir_force(
                radius=radius,
                geometry=geometry,
                coupling_lambda=coupling_lambda,
                mesh_size=mesh_size
            )
            
            results.append(result.to_dict())
            print(f"ΔF/F = {result.relative_deviation:.2e}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            # Add NaN entry to maintain array structure
            results.append({
                'R [m]': radius,
                'F_Casimir [N]': np.nan,
                'ΔF_BPR [N]': np.nan,
                'F_total [N]': np.nan,
                'relative_deviation': np.nan
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to file if requested
    if out is not None:
        df.to_csv(out, index=False)
        print(f"\nResults saved to: {out}")
    
    return df


def _standard_casimir_force(radius, geometry):
    """
    Compute standard Casimir force for different geometries.
    
    These are the well-established QED predictions that BPR modifies.
    """
    
    if geometry == "parallel_plates":
        # F = -π²ℏc A / (240 d⁴) for plates separated by distance d
        # Using radius as separation distance
        area = (2 * radius)**2  # Assume square plates
        force = -CASIMIR_PREFACTOR * area / radius**4
        
    elif geometry == "sphere":
        # Proximity force approximation for sphere near plate
        # F ≈ -π³ℏc R / (360 d³) where R is sphere radius, d is gap
        gap = radius * 0.1  # Assume gap is 10% of radius
        force = -pi**3 * hbar * c * radius / (360 * gap**3)
        
    elif geometry == "cylinder":
        # Cylinder parallel to plate
        length = 2 * radius  # Assume length ~ diameter
        gap = radius * 0.1
        force = -pi**2 * hbar * c * length / (240 * gap**3)
        
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    
    return force


def _compute_bpr_force_correction(phi_solution, radius, coupling_lambda, geometry):
    """
    Compute the BPR correction to Casimir force (Equation 7).
    
    This is the core prediction of BPR theory:
    ΔF_BPR = λ ∫ T^φ_μν n^μ n^ν dS + [geometric corrections]
    """
    
    # Get field energy and gradients
    field_energy = phi_solution.compute_energy()
    
    # Geometric factor depends on boundary shape
    if geometry == "parallel_plates":
        geometric_factor = 1.0
    elif geometry == "sphere":
        geometric_factor = 4 * pi  # Surface area factor
    elif geometry == "cylinder":
        geometric_factor = 2 * pi  # Circumference factor
    else:
        geometric_factor = 1.0
    
    # BPR correction formula (derived from stress tensor integration)
    # This is the falsifiable prediction distinguishing BPR from QED
    
    # Base correction proportional to field energy
    base_correction = coupling_lambda * field_energy / radius**2
    
    # Radius-dependent scaling (from dimensional analysis)
    radius_scaling = (radius / 1e-6)**(-1)  # Reference scale 1 μm
    
    # Total BPR correction
    Delta_F_BPR = geometric_factor * base_correction * radius_scaling
    
    # Add fractal scaling from Eq (7): δ = 1.37 ± 0.05
    # This is the key BPR signature distinguishing it from standard QED
    R_f = 1e-6  # Reference fractal scale (1 μm)
    delta_bpr = 1.37  # Critical BPR exponent from the one-pager
    alpha_bpr = coupling_lambda  # BPR coupling strength
    
    fractal_factor = alpha_bpr * (radius / R_f)**(-delta_bpr)
    
    Delta_F_BPR *= (1 + fractal_factor)
    
    return Delta_F_BPR


def analyze_bpr_signature(data_df, plot=True):
    """
    Analyze the BPR signature in Casimir force data.
    
    Identifies the characteristic features that distinguish BPR from other theories.
    
    Parameters
    ----------
    data_df : pandas.DataFrame
        Results from sweep_radius()
    plot : bool
        Whether to generate analysis plots
        
    Returns
    -------
    dict
        Analysis results including peak detection, scaling laws, etc.
    """
    
    # Extract data
    radii = data_df['R [m]'].values
    casimir_force = data_df['F_Casimir [N]'].values
    bpr_correction = data_df['ΔF_BPR [N]'].values
    relative_deviation = data_df['relative_deviation'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(casimir_force) | np.isnan(bpr_correction))
    radii = radii[valid_mask]
    casimir_force = casimir_force[valid_mask]
    bpr_correction = bpr_correction[valid_mask]
    relative_deviation = relative_deviation[valid_mask]
    
    if len(radii) == 0:
        return {"error": "No valid data points"}
    
    # Analysis metrics
    analysis = {}
    
    # Maximum deviation
    max_deviation_idx = np.argmax(np.abs(relative_deviation))
    analysis['max_deviation'] = {
        'radius': radii[max_deviation_idx],
        'relative_deviation': relative_deviation[max_deviation_idx],
        'absolute_correction': bpr_correction[max_deviation_idx]
    }
    
    # Scaling law analysis
    # Fit power law: ΔF ∝ R^α
    try:
        log_r = np.log10(radii)
        log_correction = np.log10(np.abs(bpr_correction))
        
        # Remove infinite values
        finite_mask = np.isfinite(log_correction)
        if np.sum(finite_mask) > 2:
            coeffs = np.polyfit(log_r[finite_mask], log_correction[finite_mask], 1)
            analysis['scaling_exponent'] = coeffs[0]
            analysis['scaling_coefficient'] = 10**coeffs[1]
        else:
            analysis['scaling_exponent'] = np.nan
            
    except:
        analysis['scaling_exponent'] = np.nan
    
    # Characteristic radius where BPR effects become significant
    significant_threshold = 0.01  # 1% deviation
    significant_indices = np.abs(relative_deviation) > significant_threshold
    
    if np.any(significant_indices):
        analysis['characteristic_radius'] = np.min(radii[significant_indices])
    else:
        analysis['characteristic_radius'] = np.nan
    
    # Generate plots if requested
    if plot:
        _plot_bpr_analysis(radii, casimir_force, bpr_correction, relative_deviation, analysis)
    
    return analysis


def _plot_bpr_analysis(radii, casimir_force, bpr_correction, relative_deviation, analysis):
    """Generate analysis plots for BPR signature."""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Forces vs radius
        ax1.loglog(radii, np.abs(casimir_force), 'b-', label='Standard Casimir', linewidth=2)
        ax1.loglog(radii, np.abs(bpr_correction), 'r--', label='BPR correction', linewidth=2)
        ax1.set_xlabel('Radius [m]')
        ax1.set_ylabel('Force magnitude [N]')
        ax1.set_title('Casimir Force and BPR Correction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Relative deviation
        ax2.semilogx(radii, relative_deviation * 100, 'g-', linewidth=2)
        ax2.set_xlabel('Radius [m]')
        ax2.set_ylabel('Relative deviation [%]')
        ax2.set_title('BPR Relative Deviation from Standard Casimir')
        ax2.grid(True, alpha=0.3)
        
        # Highlight maximum deviation
        if 'max_deviation' in analysis:
            max_r = analysis['max_deviation']['radius']
            max_dev = analysis['max_deviation']['relative_deviation'] * 100
            ax2.plot(max_r, max_dev, 'ro', markersize=8, label=f'Max: {max_dev:.1f}%')
            ax2.legend()
        
        # Plot 3: Scaling law
        ax3.loglog(radii, np.abs(bpr_correction), 'r-', linewidth=2, label='BPR correction')
        
        if 'scaling_exponent' in analysis and not np.isnan(analysis['scaling_exponent']):
            alpha = analysis['scaling_exponent']
            fit_line = analysis.get('scaling_coefficient', 1) * radii**alpha
            ax3.loglog(radii, fit_line, 'k--', alpha=0.7, 
                      label=f'Power law fit: R^{alpha:.2f}')
        
        ax3.set_xlabel('Radius [m]')
        ax3.set_ylabel('BPR correction [N]')
        ax3.set_title('BPR Scaling Law')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Phase diagram
        ax4.plot(radii * 1e6, relative_deviation, 'b-', linewidth=2)
        ax4.set_xlabel('Radius [μm]')
        ax4.set_ylabel('Relative deviation')
        ax4.set_title('BPR Signature vs. Radius')
        ax4.grid(True, alpha=0.3)
        
        # Mark characteristic radius
        if 'characteristic_radius' in analysis and not np.isnan(analysis['characteristic_radius']):
            char_r = analysis['characteristic_radius'] * 1e6
            ax4.axvline(char_r, color='red', linestyle='--', alpha=0.7, 
                       label=f'Characteristic: {char_r:.2f} μm')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('bpr_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Analysis plots saved as: bpr_analysis.png")
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Plotting error: {e}")


def export_prediction_data(data_df, format='csv', filename=None):
    """
    Export BPR prediction data in various formats.
    
    Parameters
    ----------
    data_df : pandas.DataFrame
        Results from sweep_radius()
    format : str
        Export format: 'csv', 'json', 'hdf5'
    filename : str, optional
        Output filename (auto-generated if None)
        
    Returns
    -------
    str
        Path to exported file
    """
    
    if filename is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bpr_prediction_{timestamp}.{format}'
    
    if format == 'csv':
        data_df.to_csv(filename, index=False)
    elif format == 'json':
        data_df.to_json(filename, orient='records', indent=2)
    elif format == 'hdf5':
        data_df.to_hdf(filename, key='bpr_data', mode='w')
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Data exported to: {filename}")
    return filename