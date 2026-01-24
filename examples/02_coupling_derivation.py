#!/usr/bin/env python3
"""
Tutorial 2: Coupling Constant Derivation

This tutorial demonstrates how BPR derives coupling constants
from substrate properties - with ZERO free parameters.

Key results:
- Gravitational: λ_grav ~ 10^-90 (91 orders below detection)
- Electromagnetic: λ_EM ~ 10^-54 (50 orders below)
- Phonon collective: λ_phonon ~ 10^-8 (potentially testable!)
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, repo_root)

from bpr.rpst.boundary_energy import (
    casimir_substrate_params,
    derive_all_couplings,
    L_PLANCK
)

# =============================================================================
# 1. Gravitational Coupling Derivation
# =============================================================================

print("=" * 60)
print("TUTORIAL 2: COUPLING CONSTANT DERIVATION")
print("=" * 60)

print("\n" + "=" * 60)
print("GRAVITATIONAL COUPLING")
print("=" * 60)

# Get default Casimir substrate parameters
params = casimir_substrate_params()

print(f"\nSubstrate parameters:")
print(f"  Prime modulus p = {params.p}")
print(f"  Number of nodes N = {params.N}")
print(f"  Coupling J = {params.J:.2e} J ({params.J/1.6e-19:.1f} eV)")
print(f"  Boundary radius R = {params.radius*100:.1f} cm")

# Derive all couplings
couplings = derive_all_couplings(params)

print(f"\nDerived parameters:")
print(f"  Stiffness κ = {couplings.kappa:.4f}")
print(f"  Correlation length ξ = {couplings.xi*1e3:.2f} mm")
print(f"  Planck length ℓ_P = {L_PLANCK:.2e} m")

print(f"\nGravitational coupling constant:")
print(f"  λ_grav = (ℓ_P²/8π) × κ × J")
print(f"        = ({L_PLANCK**2:.2e}) × {couplings.kappa:.2f} × {params.J:.2e}")
print(f"        = {couplings.lambda_bpr:.2e} J·m²")

# =============================================================================
# 2. Why Planck Suppression?
# =============================================================================

print("\n" + "=" * 60)
print("WHY PLANCK SUPPRESSION?")
print("=" * 60)

print("""
The ℓ_P² factor is UNAVOIDABLE when coupling through gravity.

The derivation chain:
  Boundary stress-energy T^μν
       ↓
  Einstein equations: G^μν = (8πG/c⁴) T^μν
       ↓
  G = ℓ_P² c³/ℏ appears
       ↓
  Coupling λ ∝ ℓ_P²

This is not a failure - it's the honest result of first-principles
derivation. ANY theory coupling boundaries to geometry through
stress-energy will have Planck suppression.
""")

# =============================================================================
# 3. Casimir Force Prediction
# =============================================================================

print("=" * 60)
print("CASIMIR FORCE PREDICTION")
print("=" * 60)

from bpr.rpst.casimir_prediction import (
    compute_bpr_casimir_prediction,
    standard_casimir_force,
    compare_to_experimental_precision
)

# Compute at 100 nm separation
separation = 100e-9  # 100 nm
pred = compute_bpr_casimir_prediction(params, separation)

print(f"\nAt separation a = {separation*1e9:.0f} nm:")
print(f"  Standard Casimir force: F/A = {pred.F_standard:.1f} Pa")
print(f"  BPR correction: δF/F = {pred.delta_F_over_F:.2e}")

comparison = compare_to_experimental_precision(pred)
print(f"\nComparison to experiment:")
print(f"  Current precision: 10^-3")
print(f"  BPR prediction: {pred.delta_F_over_F:.2e}")
print(f"  Orders below detection: {comparison['orders_below']:.0f}")

# =============================================================================
# 4. Electromagnetic Coupling
# =============================================================================

print("\n" + "=" * 60)
print("ELECTROMAGNETIC COUPLING")
print("=" * 60)

from bpr.direct_coupling.em_casimir_corrected import (
    compute_em_casimir_correction,
    critical_field
)

E_crit = critical_field()
print(f"\nSchwinger critical field: E_crit = {E_crit:.2e} V/m")

# Compute EM correction at different length scales
for scale in ['bohr', 'compton', 'planck']:
    result = compute_em_casimir_correction(length_scale=scale)
    print(f"\nEM coupling at {scale} scale:")
    print(f"  δε/ε = {result.delta_epsilon_over_epsilon:.2e}")
    print(f"  Gap to detection: {result.orders_below_precision:.0f} orders")

print("""
The electromagnetic channel is better than gravitational:
  Gravitational: 91 orders below
  EM (Compton):  50 orders below
  Improvement:   41 orders

But still not testable via direct vacuum polarization.
""")

# =============================================================================
# 5. Collective Mode Enhancement
# =============================================================================

print("=" * 60)
print("COLLECTIVE MODE ENHANCEMENT")
print("=" * 60)

from bpr.direct_coupling.collective_modes import (
    phonon_parameters,
    coupling_to_collective_mode,
    thermal_phase_gradient
)
import scipy.constants as const

# Thermal gradient
T = 300  # K
J = const.e  # 1 eV
a = 1e-9  # 1 nm
grad_phi = thermal_phase_gradient(T, J, a)

print(f"\nThermal phase gradient at room temperature:")
print(f"  |∇φ| = {grad_phi:.2e} rad/m")

# Phonon parameters
params_phonon = phonon_parameters('diamond')
print(f"\nDiamond phonon parameters:")
print(f"  Debye energy: {params_phonon.characteristic_energy/const.e*1e3:.0f} meV")
print(f"  Quality factor: {params_phonon.quality_factor:.0e}")

# Compute coupling
result = coupling_to_collective_mode(grad_phi, params_phonon, a)
print(f"\nPhonon coupling result:")
print(f"  Coupling strength: {result.coupling_strength:.2e}")
print(f"  Enhancement over QED: {result.enhancement_over_qed:.2e}×")

# =============================================================================
# 6. Stacked Enhancement
# =============================================================================

print("\n" + "=" * 60)
print("STACKED ENHANCEMENT")
print("=" * 60)

from bpr.direct_coupling.stacked_enhancement import (
    get_enhancement_catalog,
    stack_enhancements,
    compute_base_coupling
)

base = compute_base_coupling()
catalog = get_enhancement_catalog()

print(f"\nBase coupling (EM at Compton): {base:.2e}")
print(f"\nEnhancement factors available:")
for name, factor in catalog.items():
    print(f"  {factor.name}: {factor.factor:.0e}")

# Stack compatible enhancements
factors = ['coherent_1um', 'phonon', 'Q_mems']
result = stack_enhancements(factors, catalog)

print(f"\nStacking: {' + '.join(factors)}")
print(f"  Total enhancement: {result.total_enhancement:.2e}")
print(f"  Final coupling: {result.final_coupling:.2e}")
print(f"  Detection threshold: 10^-6")
print(f"  Gap: {result.gap_to_detection:.0f} orders")
print(f"  Testable: {result.is_testable}")

# =============================================================================
# 7. Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: THE PATH TO TESTABILITY")
print("=" * 60)

print("""
Channel              Gap to 10^-6    Status
─────────────────────────────────────────────
Gravitational        91 orders       Planck suppressed
EM vacuum            50 orders       Schwinger suppressed
Phonon (direct)      ~25 orders      Mode enhancement
Phonon + coherent    ~10 orders      N² enhancement
Phonon + coh + Q     1-2 orders      POTENTIALLY TESTABLE

The collective mode channel brings BPR within experimental reach!
""")
