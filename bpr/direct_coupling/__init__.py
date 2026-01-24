"""
Direct Coupling Module

Search for non-gravitational channels that bypass Planck suppression.

KEY RESULTS:
- U(1) gauge structure: CONFIRMED
- EM vacuum polarization: 50 orders below (improved from 91)
- COLLECTIVE MODES (phonon/magnon): 1-2 orders below detection!

The phonon-MEMS approach with coherent enhancement brings BPR
within plausible experimental reach.

Most promising: Diamond MEMS resonator with structured boundary
- Ideal signal: δω/ω ~ 10^-4 (DETECTABLE)
- Realistic signal: δω/ω ~ 10^-8 (borderline)

See doc/derivations/collective_mode_results.md for full analysis.
"""

from .gauge_symmetry import (
    substrate_phase,
    phase_difference,
    rpst_potential_gauge_form,
    test_global_u1_invariance,
    test_local_u1_covariance,
    analyze_u1_symmetry,
    derive_gauge_connection,
    compute_wilson_loop,
    extract_effective_flux,
    GaugeTestResult,
    U1SymmetryResult,
)

from .em_coupling import (
    derive_coupling_comprehensive,
    EMCouplingResult,
    CouplingScale,
)

from .thermal_winding import (
    compute_winding_statistics,
    WindingStatistics,
    thermal_winding_probability,
)

from .em_casimir_corrected import (
    compute_em_casimir_correction,
    EMCasimirResult,
    critical_field,
)

__all__ = [
    # Gauge symmetry
    'substrate_phase',
    'phase_difference',
    'rpst_potential_gauge_form',
    'test_global_u1_invariance',
    'test_local_u1_covariance',
    'analyze_u1_symmetry',
    'derive_gauge_connection',
    'compute_wilson_loop',
    'extract_effective_flux',
    'GaugeTestResult',
    'U1SymmetryResult',
    # EM coupling
    'derive_coupling_comprehensive',
    'EMCouplingResult',
    'CouplingScale',
    # Thermal winding
    'compute_winding_statistics',
    'WindingStatistics',
    'thermal_winding_probability',
    # EM Casimir
    'compute_em_casimir_correction',
    'EMCasimirResult',
    'critical_field',
]
