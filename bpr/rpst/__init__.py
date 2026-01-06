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
]


