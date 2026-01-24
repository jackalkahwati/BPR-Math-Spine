"""
Combined BPR Casimir Prediction - Parameter-Free

This module combines all derived parameters to generate a
complete, parameter-free prediction for Casimir force modifications.

Formula:
    ΔF/F = g · λ · exp(-a/ξ) · cos(2πa/Λ)

Where ALL parameters are derived from first principles:
- λ: From boundary energy density (boundary_energy.py)
- g: From vacuum-boundary coupling (vacuum_coupling.py)
- ξ: From correlation length (decay_oscillation.py)
- Λ: From eigenmode spacing (decay_oscillation.py)

SPRINT STATUS: Week 4 - Combined Prediction
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import json

from .boundary_energy import (
    SubstrateParameters,
    LatticeGeometry,
    derive_all_couplings,
    L_PLANCK,
    HBAR,
    C
)
from .vacuum_coupling import (
    compute_g,
    Geometry,
    VacuumCouplingResult
)
from .decay_oscillation import (
    derive_all_decay_oscillation,
    DecayOscillationParams
)


@dataclass
class CasimirPrediction:
    """Complete BPR Casimir prediction with all derived parameters."""

    # Derived parameters
    lambda_bpr: float       # Stress-energy coupling [J·m²]
    g: float                # Vacuum-boundary coupling [dimensionless]
    xi: float               # Decay length [m]
    Lambda: float           # Oscillation wavelength [m]

    # Standard Casimir
    F_standard: float       # Standard Casimir force/area [N/m²]

    # BPR correction
    delta_F_over_F: float   # Fractional correction
    delta_F: float          # Absolute correction [N/m²]

    # Component factors
    exp_factor: float       # exp(-a/ξ)
    osc_factor: float       # cos(2πa/Λ)

    # Metadata
    separation: float       # Plate separation [m]
    geometry: str           # Geometry type


def standard_casimir_force(separation: float) -> float:
    """
    Compute standard Casimir force per unit area for parallel plates.

    F/A = -π²ℏc / (240 a⁴)

    Parameters
    ----------
    separation : float
        Plate separation [m]

    Returns
    -------
    float
        Force per unit area [N/m²] (negative = attractive)
    """
    return -np.pi**2 * HBAR * C / (240 * separation**4)


def compute_bpr_casimir_prediction(
    substrate_params: SubstrateParameters,
    separation: float,
    plate_size: float = 0.01
) -> CasimirPrediction:
    """
    Compute complete BPR Casimir prediction.

    This is the main function that combines all derivations.

    Parameters
    ----------
    substrate_params : SubstrateParameters
        RPST substrate configuration
    separation : float
        Casimir gap [m]
    plate_size : float
        Characteristic plate size [m]

    Returns
    -------
    CasimirPrediction
        Complete prediction with all parameters
    """
    # 1. Derive λ from boundary energy
    couplings = derive_all_couplings(substrate_params)
    lambda_bpr = couplings.lambda_bpr

    # 2. Derive g from vacuum coupling
    # Map LatticeGeometry to Geometry
    if substrate_params.geometry == LatticeGeometry.SQUARE:
        geom = Geometry.PARALLEL_PLATES
    elif substrate_params.geometry == LatticeGeometry.SPHERE:
        geom = Geometry.SPHERE_PLATE
    else:
        geom = Geometry.PARALLEL_PLATES

    g_result = compute_g(geom, separation, plate_size)
    g = g_result.g

    # 3. Derive ξ and Λ
    decay_osc = derive_all_decay_oscillation(substrate_params, separation)
    xi = decay_osc.xi
    Lambda = decay_osc.Lambda

    # 4. Compute factors
    exp_factor = np.exp(-separation / xi)
    osc_factor = np.cos(2 * np.pi * separation / Lambda)

    # 5. Compute standard Casimir
    F_standard = standard_casimir_force(separation)

    # 6. Compute BPR correction
    # ΔF/F = g · λ · exp(-a/ξ) · cos(2πa/Λ)
    # Need to make dimensionally consistent
    # λ has units [J·m²], we need dimensionless ratio

    # The proper formula should be:
    # ΔF/F = (λ / E_char) · g · exp(-a/ξ) · cos(2πa/Λ)
    # where E_char is a characteristic energy scale

    # For Casimir, E_char ~ |F_standard| × a² = ℏc/a²
    E_char = HBAR * C / separation**2

    delta_F_over_F = (lambda_bpr / E_char) * g * exp_factor * osc_factor
    delta_F = delta_F_over_F * abs(F_standard)

    return CasimirPrediction(
        lambda_bpr=lambda_bpr,
        g=g,
        xi=xi,
        Lambda=Lambda,
        F_standard=F_standard,
        delta_F_over_F=delta_F_over_F,
        delta_F=delta_F,
        exp_factor=exp_factor,
        osc_factor=osc_factor,
        separation=separation,
        geometry=str(substrate_params.geometry.value)
    )


def generate_prediction_curve(
    substrate_params: SubstrateParameters,
    separations: np.ndarray,
    plate_size: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate BPR Casimir prediction curve.

    Parameters
    ----------
    substrate_params : SubstrateParameters
        Substrate configuration
    separations : np.ndarray
        Array of separations [m]
    plate_size : float
        Plate size [m]

    Returns
    -------
    tuple
        (separations, delta_F_over_F, F_standard)
    """
    delta_F_over_F = np.zeros_like(separations)
    F_standard = np.zeros_like(separations)

    for i, a in enumerate(separations):
        pred = compute_bpr_casimir_prediction(substrate_params, a, plate_size)
        delta_F_over_F[i] = pred.delta_F_over_F
        F_standard[i] = pred.F_standard

    return separations, delta_F_over_F, F_standard


def compare_to_experimental_precision(
    prediction: CasimirPrediction,
    experimental_precision: float = 1e-3
) -> dict:
    """
    Compare prediction to experimental precision.

    Parameters
    ----------
    prediction : CasimirPrediction
        BPR prediction
    experimental_precision : float
        Current experimental precision (default 10^-3)

    Returns
    -------
    dict
        Comparison results
    """
    ratio = abs(prediction.delta_F_over_F) / experimental_precision

    return {
        'predicted_correction': prediction.delta_F_over_F,
        'experimental_precision': experimental_precision,
        'ratio_to_precision': ratio,
        'detectable': ratio > 1,
        'orders_below': np.log10(1/ratio) if ratio > 0 else np.inf
    }


def export_prediction_data(
    substrate_params: SubstrateParameters,
    filename: str = 'bpr_casimir_prediction.json',
    n_points: int = 100,
    a_min: float = 10e-9,
    a_max: float = 1e-6
):
    """
    Export prediction data to JSON file.

    Parameters
    ----------
    substrate_params : SubstrateParameters
        Substrate configuration
    filename : str
        Output filename
    n_points : int
        Number of points
    a_min, a_max : float
        Separation range [m]
    """
    separations = np.logspace(np.log10(a_min), np.log10(a_max), n_points)
    _, delta_F_over_F, F_standard = generate_prediction_curve(
        substrate_params, separations
    )

    data = {
        'metadata': {
            'description': 'BPR Casimir prediction - parameter-free',
            'substrate_p': substrate_params.p,
            'substrate_N': substrate_params.N,
            'substrate_J_eV': substrate_params.J / 1.602e-19,
            'substrate_geometry': substrate_params.geometry.value,
            'substrate_radius_m': substrate_params.radius,
        },
        'data': {
            'separation_m': separations.tolist(),
            'delta_F_over_F': delta_F_over_F.tolist(),
            'F_standard_Pa': F_standard.tolist(),
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Prediction data exported to {filename}")


def summary_report(substrate_params: SubstrateParameters, separation: float) -> str:
    """
    Generate human-readable summary report.

    Parameters
    ----------
    substrate_params : SubstrateParameters
        Substrate configuration
    separation : float
        Typical separation [m]

    Returns
    -------
    str
        Formatted report
    """
    pred = compute_bpr_casimir_prediction(substrate_params, separation)
    comparison = compare_to_experimental_precision(pred)

    lines = [
        "=" * 70,
        "BPR CASIMIR PREDICTION - PARAMETER-FREE DERIVATION",
        "=" * 70,
        "",
        "SUBSTRATE PARAMETERS:",
        f"  Prime modulus p = {substrate_params.p}",
        f"  Number of nodes N = {substrate_params.N}",
        f"  Coupling J = {substrate_params.J:.3e} J = {substrate_params.J/1.602e-19:.2f} eV",
        f"  Geometry = {substrate_params.geometry.value}",
        f"  Characteristic size = {substrate_params.radius*100:.1f} cm",
        "",
        "DERIVED PARAMETERS:",
        f"  λ_BPR = {pred.lambda_bpr:.3e} J·m²",
        f"  g = {pred.g:.3e}",
        f"  ξ = {pred.xi:.3e} m = {pred.xi*1e6:.1f} μm",
        f"  Λ = {pred.Lambda:.3e} m = {pred.Lambda*100:.1f} cm",
        "",
        f"CASIMIR SEPARATION: {separation*1e9:.0f} nm",
        "",
        "COMPONENT FACTORS:",
        f"  exp(-a/ξ) = {pred.exp_factor:.6f}",
        f"  cos(2πa/Λ) = {pred.osc_factor:.6f}",
        "",
        "PREDICTION:",
        f"  Standard Casimir F/A = {pred.F_standard:.3e} N/m²",
        f"  BPR correction ΔF/F = {pred.delta_F_over_F:.3e}",
        f"  Absolute correction = {pred.delta_F:.3e} N/m²",
        "",
        "COMPARISON TO EXPERIMENT:",
        f"  Current precision = {comparison['experimental_precision']:.0e}",
        f"  Prediction / precision = {comparison['ratio_to_precision']:.3e}",
        f"  Orders of magnitude below = {comparison['orders_below']:.1f}",
        f"  Detectable: {'YES' if comparison['detectable'] else 'NO'}",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    from .boundary_energy import casimir_substrate_params

    # Create default parameters
    params = casimir_substrate_params()

    # Generate report
    report = summary_report(params, separation=100e-9)
    print(report)

    # Generate prediction curve
    print("\nGenerating prediction curve...")
    seps = np.logspace(-8, -6, 20)  # 10nm to 1μm
    _, corrections, _ = generate_prediction_curve(params, seps)

    print(f"\n{'Separation (nm)':<20} {'ΔF/F':<15}")
    print("-" * 35)
    for a, dF in zip(seps, corrections):
        print(f"{a*1e9:<20.1f} {dF:<15.3e}")
