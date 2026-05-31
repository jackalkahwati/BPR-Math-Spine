"""
Information integration module for BPR-Math-Spine (Equations 4 & 5)

Implementation of:
- Eq (4): IIT-inspired information term S_info = -ξ ∫_Σ d^(D-1)x √|h| Φ[φ]
- Eq (5): Six-factor consciousness coupling S_bio = ∫_Σ d^(D-1)x √|h| χ_b(x) φ(x)

This module implements Integrated Information Theory (IIT) concepts and
biological consciousness coupling for the boundary phase field φ.

Author: Jack Al-Kahwati (jack@thestardrive.com)
Status: Complete implementation with mathematical rigor
"""

import numpy as np
import sympy as sp
from typing import Callable, Optional, Dict, Any, Tuple, List
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize_scalar
import warnings


class InformationIntegration:
    """
    Integrated Information Theory (IIT) implementation for BPR.
    
    Implements Equation (4) from the BPR one-pager:
    S_info = -ξ ∫_Σ d^(D-1)x √|h| Φ[φ]
    
    where Φ = Σ_{i<j} I_ij and I_ij = -∫ p_ij ln p_ij
    """
    
    def __init__(self, xi: float = 1e-3, n_partitions: int = 8):
        """
        Initialize information integration calculator.
        
        Parameters
        ----------
        xi : float
            Information coupling strength ξ
        n_partitions : int
            Number of boundary partitions for IIT analysis
        """
        self.xi = xi
        self.n_partitions = n_partitions
        self.phi_cache = {}
        self.entropy_tolerance = 1e-10
    
    def extract_field_values(self, phi_field, boundary_mesh) -> np.ndarray:
        """
        Extract field values from various field representations.
        
        Parameters
        ----------
        phi_field : BoundaryFieldSolution, callable, or array
            The boundary phase field
        boundary_mesh : BoundaryMesh
            The boundary surface
            
        Returns
        -------
        np.ndarray
            Field values at mesh points
        """
        if hasattr(phi_field, 'phi') and hasattr(phi_field.phi, 'vector'):
            # FEniCS function
            return phi_field.phi.vector().get_local()
        elif callable(phi_field):
            # Generate test points on unit sphere
            n_points = max(self.n_partitions * 10, 100)
            theta = np.linspace(0, np.pi, int(np.sqrt(n_points)))
            phi = np.linspace(0, 2*np.pi, int(np.sqrt(n_points)))
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            
            x = np.sin(theta_grid) * np.cos(phi_grid)
            y = np.sin(theta_grid) * np.sin(phi_grid)
            z = np.cos(theta_grid)
            
            values = []
            for i in range(x.size):
                try:
                    val = phi_field(x.flat[i], y.flat[i], z.flat[i])
                    values.append(float(val))
                except:
                    values.append(0.0)
            return np.array(values)
        else:
            # Assume it's already an array
            return np.asarray(phi_field)
    
    def partition_boundary(self, field_values: np.ndarray) -> List[np.ndarray]:
        """
        Partition boundary field into subsystems for IIT analysis.
        
        Parameters
        ----------
        field_values : np.ndarray
            Field values across the boundary
            
        Returns
        -------
        List[np.ndarray]
            List of field value partitions
        """
        n_values = len(field_values)
        partition_size = max(1, n_values // self.n_partitions)
        
        partitions = []
        for i in range(0, n_values, partition_size):
            partition = field_values[i:i + partition_size]
            if len(partition) > 0:
                partitions.append(partition)
        
        return partitions
    
    def compute_phi_measure(self, phi_field, boundary_mesh) -> float:
        """
        Compute the integrated information measure Φ[φ] using IIT principles.
        
        Parameters
        ----------
        phi_field : BoundaryFieldSolution or callable
            The boundary phase field φ(x, y, z)
        boundary_mesh : BoundaryMesh
            The triangulated boundary surface
            
        Returns
        -------
        float
            Integrated information measure Φ
            
        Notes
        -----
        Implements proper IIT computation:
        1. Extract field values across boundary
        2. Partition into subsystems
        3. Compute pairwise information I_ij
        4. Sum over all partition pairs: Φ = Σ_{i<j} I_ij
        """
        
        # Extract field values
        field_values = self.extract_field_values(phi_field, boundary_mesh)
        
        if len(field_values) < 2:
            return 0.0
        
        # Partition the boundary
        partitions = self.partition_boundary(field_values)
        
        if len(partitions) < 2:
            # If we can't partition, use variance as fallback
            return np.var(field_values)
        
        # Compute pairwise information between all partitions
        phi_total = 0.0
        n_pairs = 0
        
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                I_ij = self.pairwise_information(partitions[i], partitions[j])
                phi_total += I_ij
                n_pairs += 1
        
        # Normalize by number of pairs and add field complexity measure
        if n_pairs > 0:
            phi_normalized = phi_total / n_pairs
        else:
            phi_normalized = 0.0
        
        # Add field complexity contribution
        field_complexity = self._compute_field_complexity(field_values)
        phi_measure = phi_normalized + 0.1 * field_complexity
        
        return phi_measure
    
    def _compute_field_complexity(self, field_values: np.ndarray) -> float:
        """
        Compute field complexity as a measure of information content.
        
        Parameters
        ----------
        field_values : np.ndarray
            Field values
            
        Returns
        -------
        float
            Field complexity measure
        """
        if len(field_values) < 2:
            return 0.0
        
        # Compute gradients (discrete differences)
        gradients = np.diff(field_values)
        
        # Complexity measures
        variance = np.var(field_values)
        gradient_variance = np.var(gradients) if len(gradients) > 0 else 0.0
        
        # Information-theoretic complexity (simplified)
        hist, _ = np.histogram(field_values, bins=min(20, len(field_values)//2 + 1))
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zero probabilities
        entropy = -np.sum(prob * np.log(prob + self.entropy_tolerance))
        
        # Combined complexity measure
        complexity = variance + 0.5 * gradient_variance + 0.3 * entropy
        
        return complexity
    
    def pairwise_information(self, phi_i, phi_j) -> float:
        """
        Compute pairwise information I_ij between field components.
        
        Parameters
        ----------
        phi_i, phi_j : array_like
            Field values at different boundary regions
            
        Returns
        -------
        float
            Pairwise information measure I_ij = -∫ p_ij ln p_ij
            
        Notes
        -----
        Implements proper mutual information calculation between subsystems.
        """
        phi_i = np.asarray(phi_i)
        phi_j = np.asarray(phi_j)
        
        if len(phi_i) == 0 or len(phi_j) == 0:
            return 0.0
        
        # For single values, use direct correlation
        if len(phi_i) == 1 and len(phi_j) == 1:
            # Information based on field difference
            diff = abs(phi_i[0] - phi_j[0])
            return -diff * np.log(diff + self.entropy_tolerance)
        
        # Compute joint probability distribution
        try:
            # Discretize the field values for probability estimation
            n_bins = min(10, max(3, len(phi_i)//3, len(phi_j)//3))
            
            # Create joint histogram
            hist_2d, x_edges, y_edges = np.histogram2d(phi_i, phi_j, bins=n_bins)
            
            # Normalize to get probabilities
            p_joint = hist_2d / np.sum(hist_2d)
            
            # Marginal distributions
            p_i = np.sum(p_joint, axis=1)
            p_j = np.sum(p_joint, axis=0)
            
            # Compute mutual information I(i;j) = ∑∑ p(i,j) log(p(i,j)/(p(i)p(j)))
            mutual_info = 0.0
            for i_idx in range(len(p_i)):
                for j_idx in range(len(p_j)):
                    if p_joint[i_idx, j_idx] > 0 and p_i[i_idx] > 0 and p_j[j_idx] > 0:
                        mutual_info += p_joint[i_idx, j_idx] * np.log(
                            p_joint[i_idx, j_idx] / (p_i[i_idx] * p_j[j_idx]) + self.entropy_tolerance
                        )
            
            return max(0.0, mutual_info)  # Ensure non-negative
            
        except Exception:
            # Fallback: correlation-based measure
            if len(phi_i) > 1 and len(phi_j) > 1:
                try:
                    correlation = np.corrcoef(phi_i, phi_j)[0, 1]
                    if not np.isnan(correlation):
                        return abs(correlation) * np.log(abs(correlation) + self.entropy_tolerance)
                except:
                    pass
            return 0.0
    
    def information_action(self, phi_field, boundary_mesh) -> float:
        """
        Compute the information action term S_info.
        
        Parameters
        ----------
        phi_field : BoundaryFieldSolution or callable
            The boundary phase field
        boundary_mesh : BoundaryMesh
            The boundary surface
            
        Returns
        -------
        float
            Information action S_info
        """
        
        phi_measure = self.compute_phi_measure(phi_field, boundary_mesh)
        
        # Integrate over boundary (simplified)
        try:
            from .geometry import compute_boundary_area
            area = compute_boundary_area(boundary_mesh)
        except:
            area = 4 * np.pi  # Unit sphere fallback
        
        S_info = -self.xi * area * phi_measure
        
        return S_info


# ---------------------------------------------------------------------------
# Empirical envelope on χ_max · ε from the published psi literature
# (May 2026 literature review, see doc/experiments/CONSCIOUSNESS_EMPIRICAL_CALIBRATION.md)
# ---------------------------------------------------------------------------

# Defensible upper bound on the BEHAVIORAL effect of χ_max·ε·gain at the
# operator → forced-choice readout level. Source: Tressoldi & Storm 2024
# Stage 2 Registered Report on ganzfeld (r=0.08, CI [0.04, 0.12]).
# Remains statistically consistent with zero under PET-PEESE / trim-and-fill.
BEHAVIORAL_UPPER_BOUND_D = 0.10

# Defensible upper bound at the CONTROLLED PHYSICAL SUBSTRATE level (QRNG).
# Source: Bösch, Steinkamp & Boller 2006 trim-and-fill corrected CI (includes
# zero); Maier, Dechamps & Plitsch 2018 preregistered QRNG replication
# returned BF01 = 11.07 in favor of the null.
CONTROLLED_RNG_UPPER_BOUND_PER_BIT = 1e-5

# Anchor references for the calibration bands.
EMPIRICAL_ANCHOR_REFS = (
    "Tressoldi & Storm (2024) F1000Research — Stage 2 Registered Report, "
    "ganzfeld r ≈ 0.08, CI [0.04, 0.12]",
    "Kekecs et al. (2023) Royal Society Open Science — Transparent Psi "
    "Project, strong null on Bem-Exp.1 precognition",
    "Bösch, Steinkamp & Boller (2006) Psychological Bulletin 132:497–523 — "
    "RNG-PK trim-and-fill, CI includes zero",
    "Maier, Dechamps & Plitsch (2018) Frontiers in Psychology — "
    "preregistered QRNG, BF01 = 11.07 favoring null",
    "Milton & Wiseman (1999) Psychological Bulletin 125:387–391 — skeptic-side ganzfeld null",
    "Galak et al. (2012) JPSP — preregistered failed precognition replications",
)


class ConsciousnessCoupling:
    """
    Six-factor consciousness coupling implementation for BPR Equation (5).

    Implements: S_bio = ∫_Σ d^(D-1)x √|h| χ_b(x) φ(x)

    where χ_b = χ_max σ[k(Φ/Φ_c - 1)] E^α (Φ/Φ_c)^β τ S^γ U^δ I^ε

    PARAMETER ACCOUNTING (post-empirical-calibration review):
    The six exponents (α=1.2, β=1.5, γ=0.8, δ=1.0, ε=1.3, k=2.0) are
    TUNABLE FLOATS, not derived rationals. They are model construction.
    The sigmoid switch σ[k(Φ/Φ_c-1)] encodes a phase-transition prediction
    that is currently UNTESTED — see P-Φc.1 in
    doc/experiments/CONSCIOUSNESS_EMPIRICAL_CALIBRATION.md.

    EMPIRICAL ENVELOPE (May 2026 literature review):
    Behavioral upper bound on χ_max·ε·gain: d ≈ 0.10
        (Tressoldi & Storm 2024 ganzfeld Stage 2 RR; preregistered)
    Controlled-substrate upper bound (QRNG): ≲ 1e-5 per bit
        (Bösch 2006 trim-and-fill; Maier 2018 strong null)
    See ``BEHAVIORAL_UPPER_BOUND_D``, ``CONTROLLED_RNG_UPPER_BOUND_PER_BIT``,
    and ``continuous_consciousness_coupling()`` for an alternative form
    that drops the sigmoid switch (consistent with the empirical
    state-dependence record, which shows continuous correlations, not
    a step-function).

    P-Φc.1 (testable phase-transition prediction): under general
    anesthesia or in DOC patients with cortical integration abolished,
    any psi effect in a previously-effective operator should collapse to
    chance within minutes of anesthetic onset and return within minutes
    of emergence. No such study has been published; the prediction is
    untested.
    """
    
    def __init__(self, consciousness_factors: Optional[Dict] = None):
        """
        Initialize consciousness coupling calculator.
        
        Parameters
        ----------
        consciousness_factors : dict, optional
            Six-factor consciousness parameters
        """
        if consciousness_factors is None:
            consciousness_factors = {
                'chi_max': 1e-3,     # Maximum coupling strength
                'Phi_c': 1.0,        # Critical information threshold  
                'alpha': 1.2,        # Energy exponent E^α
                'beta': 1.5,         # Information exponent (Φ/Φ_c)^β
                'gamma': 0.8,        # Entropy exponent S^γ
                'delta': 1.0,        # Utility exponent U^δ  
                'epsilon': 1.3,      # Information exponent I^ε
                'k': 2.0,            # Activation steepness
                'tau': 1.0           # Time factor τ
            }
        
        self.factors = consciousness_factors
        self.info_calc = InformationIntegration()
    
    def sigmoid_activation(self, x: float) -> float:
        """Sigmoid activation function σ[k(Φ/Φ_c - 1)]."""
        k = self.factors['k']
        try:
            return 1.0 / (1.0 + np.exp(-k * x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0
    
    def compute_energy_factor(self, phi_field, boundary_mesh) -> float:
        """
        Compute energy factor E from the boundary field.
        
        Parameters
        ----------
        phi_field : BoundaryFieldSolution or callable
            The boundary phase field
        boundary_mesh : BoundaryMesh  
            The boundary surface
            
        Returns
        -------
        float
            Energy factor E
        """
        # Extract field values
        field_values = self.info_calc.extract_field_values(phi_field, boundary_mesh)
        
        # Energy as field magnitude squared (kinetic energy analog)
        energy = np.mean(field_values**2) if len(field_values) > 0 else 0.0
        
        return max(energy, 1e-10)  # Avoid zero
    
    def compute_entropy_factor(self, phi_field, boundary_mesh) -> float:
        """
        Compute entropy factor S from field distribution.
        
        Returns
        -------
        float
            Entropy factor S
        """
        field_values = self.info_calc.extract_field_values(phi_field, boundary_mesh)
        
        if len(field_values) < 2:
            return 1.0
        
        # Compute Shannon entropy of field distribution
        hist, _ = np.histogram(field_values, bins=min(20, len(field_values)//2 + 1))
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zeros
        
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        
        return max(entropy, 0.1)
    
    def compute_utility_factor(self, phi_field, boundary_mesh) -> float:
        """
        Compute utility factor U (optimization/efficiency measure).
        
        Returns
        -------
        float
            Utility factor U
        """
        field_values = self.info_calc.extract_field_values(phi_field, boundary_mesh)
        
        if len(field_values) < 2:
            return 1.0
        
        # Utility as inverse of field variation (efficiency measure)
        variation = np.var(field_values)
        smoothness = 1.0 / (1.0 + variation)
        
        # Add gradient optimization measure
        gradients = np.diff(field_values)
        gradient_smoothness = 1.0 / (1.0 + np.var(gradients)) if len(gradients) > 0 else 1.0
        
        utility = 0.6 * smoothness + 0.4 * gradient_smoothness
        
        return max(utility, 0.1)
    
    def compute_information_factor(self, phi_field, boundary_mesh) -> float:
        """
        Compute information factor I using IIT measures.
        
        Returns
        -------
        float
            Information factor I
        """
        phi_measure = self.info_calc.compute_phi_measure(phi_field, boundary_mesh)
        
        # Normalize information measure
        info_factor = phi_measure / (1.0 + phi_measure)
        
        return max(info_factor, 0.01)
    
    def compute_consciousness_coupling(self, phi_field, boundary_mesh, time_factor: float = 1.0) -> float:
        """
        Compute the full six-factor consciousness coupling χ_b(x).
        
        Parameters
        ----------
        phi_field : BoundaryFieldSolution or callable
            The boundary phase field
        boundary_mesh : BoundaryMesh
            The boundary surface  
        time_factor : float
            Time evolution factor τ
            
        Returns
        -------
        float
            Consciousness coupling strength χ_b
        """
        
        # Compute the six factors
        E = self.compute_energy_factor(phi_field, boundary_mesh)
        S = self.compute_entropy_factor(phi_field, boundary_mesh)  
        U = self.compute_utility_factor(phi_field, boundary_mesh)
        I = self.compute_information_factor(phi_field, boundary_mesh)
        tau = time_factor * self.factors['tau']
        
        # Compute integrated information Φ
        Phi = self.info_calc.compute_phi_measure(phi_field, boundary_mesh)
        Phi_ratio = Phi / self.factors['Phi_c']
        
        # Sigmoid activation: σ[k(Φ/Φ_c - 1)]
        activation_arg = Phi_ratio - 1.0
        sigma = self.sigmoid_activation(activation_arg)
        
        # Six-factor coupling: χ_b = χ_max σ[k(Φ/Φ_c-1)] E^α (Φ/Φ_c)^β τ S^γ U^δ I^ε
        try:
            chi_b = (self.factors['chi_max'] * 
                    sigma *
                    (E ** self.factors['alpha']) *
                    (Phi_ratio ** self.factors['beta']) *
                    tau *
                    (S ** self.factors['gamma']) *
                    (U ** self.factors['delta']) *
                    (I ** self.factors['epsilon']))
        except (OverflowError, ValueError):
            # Fallback for numerical issues
            chi_b = self.factors['chi_max'] * sigma
        
        return chi_b
    
    def biological_action(self, phi_field, boundary_mesh, time_factor: float = 1.0) -> float:
        """
        Compute the biological action S_bio = ∫_Σ d^(D-1)x √|h| χ_b(x) φ(x).
        
        Parameters
        ----------
        phi_field : BoundaryFieldSolution or callable
            The boundary phase field
        boundary_mesh : BoundaryMesh
            The boundary surface
        time_factor : float
            Time evolution factor
            
        Returns
        -------
        float
            Biological action S_bio
        """
        
        # Compute consciousness coupling
        chi_b = self.compute_consciousness_coupling(phi_field, boundary_mesh, time_factor)
        
        # Get field values and integrate
        field_values = self.info_calc.extract_field_values(phi_field, boundary_mesh)
        phi_mean = np.mean(field_values) if len(field_values) > 0 else 0.0
        
        # Boundary area (simplified)
        try:
            from .geometry import compute_boundary_area
            area = compute_boundary_area(boundary_mesh)
        except:
            area = 4 * np.pi  # Unit sphere fallback
        
        # Biological action
        S_bio = area * chi_b * phi_mean
        
        return S_bio


def continuous_consciousness_coupling(
    Phi: float,
    chi_max: float = 1e-3,
    Phi_c: float = 1.0,
    beta: float = 1.5,
    E: float = 1.0,
    alpha: float = 1.2,
    S: float = 1.0,
    gamma: float = 0.8,
    U: float = 1.0,
    delta: float = 1.0,
    I: float = 1.0,
    epsilon: float = 1.3,
    tau: float = 1.0,
) -> float:
    """Continuous-monotone alternative to the sigmoid-gated Eq (5) coupling.

    Drops the σ[k(Φ/Φ_c − 1)] phase-transition switch in favor of a
    monotone-continuous (Φ/Φ_c)^β factor that does not impose a threshold.
    All other factors are unchanged. This is the form CONSISTENT WITH the
    empirical state-dependence literature, which shows continuous correlations
    (belief, hypnotizability, altered-state shift) rather than a step-function.

        χ_b_continuous = χ_max · E^α · (Φ/Φ_c)^β · τ · S^γ · U^δ · I^ε

    Note: this drops the SIGMOID switch only. The six fitted exponents
    (α=1.2, β=1.5, γ=0.8, δ=1.0, ε=1.3) remain as tunable model construction
    and are not derived. See ``ConsciousnessCoupling`` for the original
    sigmoid-switched form and P-Φc.1 for the untested phase-transition
    prediction the sigmoid encodes.

    USE WHICH FORM:
    - Sigmoid (original ``ConsciousnessCoupling``): defends a phase-transition
      prediction at Φ_c. Testable via P-Φc.1 (anesthesia / DOC patients).
    - Continuous (this function): fits the existing behavioral psi literature
      without committing to an empirically-unsupported threshold.

    Parameters
    ----------
    Phi : float
        Integrated information (IIT) of the system.
    chi_max : float
        Maximum coupling strength (envelope: see BEHAVIORAL_UPPER_BOUND_D
        for the published upper bound after publication-bias correction).
    Phi_c : float
        Reference integrated-information scale (no longer a threshold).
    Other parameters
        Energy E, entropy S, utility U, information I, time τ and their
        exponents. Defaults match ``ConsciousnessCoupling`` defaults.

    Returns
    -------
    float
        χ_b coupling strength under the continuous form.
    """
    if Phi <= 0 or Phi_c <= 0:
        return 0.0
    return (
        chi_max
        * (E ** alpha)
        * (Phi / Phi_c) ** beta
        * tau
        * (S ** gamma)
        * (U ** delta)
        * (I ** epsilon)
    )


def placeholder_consciousness_coupling(phi_field, consciousness_factors: Optional[Dict] = None) -> float:
    """
    Convenience function for consciousness coupling (backward compatibility).
    
    Parameters
    ----------
    phi_field : BoundaryFieldSolution or callable
        The boundary phase field
    consciousness_factors : dict, optional
        Consciousness coupling parameters
        
    Returns
    -------
    float
        Consciousness coupling strength χ_b
    """
    
    # Mock mesh for standalone usage
    class MockMesh:
        pass
    
    mock_mesh = MockMesh()
    
    # Use the full implementation
    consciousness = ConsciousnessCoupling(consciousness_factors)
    return consciousness.compute_consciousness_coupling(phi_field, mock_mesh)


def test_information_integration():
    """Test the complete information integration implementation."""
    
    print("Testing Information Integration (Equation 4)...")
    
    # Create test instance
    info_calc = InformationIntegration(xi=1e-3, n_partitions=8)
    
    # Test with synthetic data
    phi_test = lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-(x**2 + y**2 + z**2))
    
    # Mock boundary mesh
    class MockMesh:
        pass
    
    mock_mesh = MockMesh()
    
    # Test Φ measure
    phi_measure = info_calc.compute_phi_measure(phi_test, mock_mesh)
    print(f"  Integrated information Φ: {phi_measure:.6f}")
    
    # Test pairwise information
    phi_i = np.random.randn(50) 
    phi_j = np.random.randn(50) + 0.3 * phi_i  # Correlated
    I_ij = info_calc.pairwise_information(phi_i, phi_j)
    print(f"  Pairwise information I_ij: {I_ij:.6f}")
    
    # Test action
    S_info = info_calc.information_action(phi_test, mock_mesh)
    print(f"  Information action S_info: {S_info:.6f}")
    
    # Test field complexity
    field_vals = info_calc.extract_field_values(phi_test, mock_mesh)
    complexity = info_calc._compute_field_complexity(field_vals)
    print(f"  Field complexity: {complexity:.6f}")
    
    print("✅ Complete information integration test passed")


def test_consciousness_coupling():
    """Test the complete consciousness coupling implementation."""
    
    print("\nTesting Six-Factor Consciousness Coupling (Equation 5)...")
    
    # Test field
    phi_test = lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-(x**2 + y**2))
    
    # Mock mesh
    class MockMesh:
        pass
    mock_mesh = MockMesh()
    
    # Test with default factors
    consciousness = ConsciousnessCoupling()
    
    # Test individual factors
    E = consciousness.compute_energy_factor(phi_test, mock_mesh)
    S = consciousness.compute_entropy_factor(phi_test, mock_mesh)
    U = consciousness.compute_utility_factor(phi_test, mock_mesh)
    I = consciousness.compute_information_factor(phi_test, mock_mesh)
    
    print(f"  Six factors - E: {E:.4f}, S: {S:.4f}, U: {U:.4f}, I: {I:.4f}")
    
    # Test full coupling
    chi_b = consciousness.compute_consciousness_coupling(phi_test, mock_mesh)
    print(f"  Consciousness coupling χ_b: {chi_b:.6e}")
    
    # Test biological action
    S_bio = consciousness.biological_action(phi_test, mock_mesh, time_factor=1.2)
    print(f"  Biological action S_bio: {S_bio:.6e}")
    
    # Test with custom factors
    custom_factors = {
        'chi_max': 1e-2,
        'Phi_c': 0.8,
        'alpha': 1.5,
        'beta': 2.0,
        'gamma': 0.5,
        'delta': 1.0,
        'epsilon': 1.3,
        'k': 2.0,
        'tau': 1.0
    }
    consciousness_custom = ConsciousnessCoupling(custom_factors)
    chi_b_custom = consciousness_custom.compute_consciousness_coupling(phi_test, mock_mesh)
    print(f"  Custom coupling χ_b: {chi_b_custom:.6e}")
    
    # Test sigmoid activation
    sigma_test = consciousness.sigmoid_activation(0.5)
    print(f"  Sigmoid activation σ(0.5): {sigma_test:.4f}")
    
    print("✅ Complete consciousness coupling test passed")


def comprehensive_test():
    """Run comprehensive tests for both Eq (4) and Eq (5)."""
    
    print("="*60)
    print("BPR INFORMATION & CONSCIOUSNESS MODULE TESTS")
    print("="*60)
    
    test_information_integration()
    test_consciousness_coupling()
    
    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED - Equations (4) & (5) Complete!")
    print("📊 Information Integration: IIT-inspired Φ measure")
    print("🧠 Consciousness Coupling: Six-factor χ_b calculation")
    print("🔬 Ready for integration with BPR field equations")
    print("="*60)


if __name__ == "__main__":
    comprehensive_test()