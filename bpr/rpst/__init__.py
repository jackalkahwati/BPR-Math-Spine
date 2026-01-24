"""
Resonant Prime Substrate Theory (RPST) - Layer 0
Microscopic foundations for Boundary Phase Resonance (BPR).

This package is intentionally self-contained and uses only NumPy/SciPy/SymPy
already present in this repo's dependencies.
"""

from .substrate import SubstrateState, PrimeField, legendre_symbol
from .dynamics import SymplecticEvolution
from .topology import compute_winding_number, TopologicalCharge
from .coarse_grain import CoarseGraining, verify_wave_equation
from .hamiltonian import RPSTHamiltonian, compute_spectral_zeta, verify_gue_level_spacing
from .coherence import (
    PhaseCoherence,
    BoundaryStress,
    EligibilityFunctional,
    CoherenceMetrics,
    StressMetrics,
    create_ring_lattice,
    create_square_lattice,
)
from .boundary_energy import (
    SubstrateParameters,
    LatticeGeometry,
    DerivedCouplings,
    derive_kappa,
    derive_all_couplings,
    casimir_substrate_params,
)
from .vacuum_coupling import (
    compute_g,
    Geometry,
    VacuumCouplingResult,
)
from .decay_oscillation import (
    derive_all_decay_oscillation,
    DecayOscillationParams,
)
from .casimir_prediction import (
    compute_bpr_casimir_prediction,
    CasimirPrediction,
    standard_casimir_force,
    compare_to_experimental_precision,
    summary_report,
)

__all__ = [
    "SubstrateState",
    "PrimeField",
    "legendre_symbol",
    "SymplecticEvolution",
    "compute_winding_number",
    "TopologicalCharge",
    "CoarseGraining",
    "verify_wave_equation",
    "RPSTHamiltonian",
    "compute_spectral_zeta",
    "verify_gue_level_spacing",
    # Coherence and stress metrics
    "PhaseCoherence",
    "BoundaryStress",
    "EligibilityFunctional",
    "CoherenceMetrics",
    "StressMetrics",
    "create_ring_lattice",
    "create_square_lattice",
    # Parameter-free Casimir derivation
    "SubstrateParameters",
    "LatticeGeometry",
    "DerivedCouplings",
    "derive_kappa",
    "derive_all_couplings",
    "casimir_substrate_params",
    "compute_g",
    "Geometry",
    "VacuumCouplingResult",
    "derive_all_decay_oscillation",
    "DecayOscillationParams",
    "compute_bpr_casimir_prediction",
    "CasimirPrediction",
    "standard_casimir_force",
    "compare_to_experimental_precision",
    "summary_report",
]


