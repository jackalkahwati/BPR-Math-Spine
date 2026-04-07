"""
Turing pattern wavelength validator
=====================================

Well dataset : ``gray_scott_reaction_diffusion``
BPR prediction: P23.1 — Turing pattern wavelength scales as
                λ_T = 2π √(D_u / |f_u|) where f_u is the activator
                self-inhibition rate at steady state.

BPR reproduces standard Turing instability theory via the morphogenetic
field equation (§11.2). This is a CONSISTENT prediction — BPR gives the
correct Turing wavelength; The Well lets us verify against PDE simulation
instead of only the CIMA experiment.

Method
------
1. Load one frame of the v-field (autocatalyst) from gray_scott.
2. 2-D FFT → radially-averaged power spectrum → peak wavenumber k̂.
3. Observed wavelength: λ_obs = 2π / k̂  (in grid units × domain size).
4. BPR prediction: λ_BPR = 2π √(D_u / (2 u* (v*)²))
   using numerical steady-state (u*, v*) for the dataset's (F, k, D_u, D_v).
5. Σ deviation = |λ_obs − λ_BPR| / (0.10 × λ_BPR)  [theory precision 10%].
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Gray-Scott parameters (The Well defaults — checked against their README)
# ---------------------------------------------------------------------------

@dataclass
class GrayScottParams:
    """Default Gray-Scott parameters used in The Well dataset."""
    D_u: float = 0.16    # diffusion coefficient of u (substrate)
    D_v: float = 0.08    # diffusion coefficient of v (activator)
    F:   float = 0.035   # feed rate
    k:   float = 0.065   # kill rate
    domain_size: float = 1.0   # simulation domain side length (dimensionless)


def _gray_scott_steady_state(p: GrayScottParams) -> tuple[float, float]:
    """Approximate non-trivial steady state (u*, v*) for Gray-Scott.

    Solves: F(1 − u) − u v² = 0  and  u v² − (F+k) v = 0
    Non-trivial branch: v* = F(1 − u*) / (u* v*) → v* = (u* − 1 − k/F) ...

    Uses the analytic expression from Pearson (1993):
        v* = (F - F√(1 − 4(F+k)²/F)) / (2(F+k))  [approximation for small F]
    Falls back to simple numerical iteration if needed.
    """
    F, k = p.F, p.k
    discriminant = 1.0 - 4.0 * (F + k) ** 2 / F
    if discriminant <= 0:
        # No non-trivial Turing-capable steady state at these parameters
        return 1.0, 0.0
    v_star = F * (1.0 - math.sqrt(discriminant)) / (2.0 * (F + k))
    u_star = (F + k) / (F / (v_star + 1e-12) * v_star) if v_star > 0 else 1.0
    # clamp to physical range
    u_star = float(np.clip(u_star, 0.01, 1.0))
    v_star = float(np.clip(v_star, 0.0, 1.0))
    return u_star, v_star


def bpr_turing_wavelength(params: Optional[GrayScottParams] = None) -> float:
    """BPR prediction for Turing pattern wavelength (in domain units).

    λ_T = 2π √(D_u / |f_u|)

    where f_u = ∂(du/dt)/∂u|* = −v*² − F  (linearised Gray-Scott Jacobian).
    """
    p = params or GrayScottParams()
    u_star, v_star = _gray_scott_steady_state(p)
    f_u = -(v_star ** 2) - p.F   # always negative at steady state
    lambda_bpr = 2.0 * math.pi * math.sqrt(p.D_u / abs(f_u))
    return lambda_bpr


def _radial_power_spectrum(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum of a 2-D field.

    Returns (k_radial, power) where k is in units of 2π/N_grid.
    """
    ny, nx = field_2d.shape
    fft2 = np.fft.fft2(field_2d)
    power = (np.abs(fft2) ** 2)
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_max = int(min(nx, ny) / 2)
    k_bins = np.arange(1, k_max)
    power_radial = np.zeros(len(k_bins))
    for i, kb in enumerate(k_bins):
        mask = (K >= kb - 0.5) & (K < kb + 0.5)
        if mask.any():
            power_radial[i] = power[mask].mean()
    return k_bins.astype(float), power_radial


def measure_turing_wavelength(v_field: np.ndarray,
                               domain_size: float = 1.0) -> float:
    """Extract dominant Turing pattern wavelength from a Gray-Scott v-field.

    Parameters
    ----------
    v_field : ndarray of shape (ny, nx)
    domain_size : float
        Physical domain side length (same units as desired output).

    Returns
    -------
    float : dominant pattern wavelength in domain units.
    """
    if v_field.ndim > 2:
        # Take last 2-D spatial slice
        v_field = v_field.reshape(-1, v_field.shape[-2], v_field.shape[-1])[0]

    ks, pw = _radial_power_spectrum(v_field)
    # Ignore k=0 and low-frequency bias; find peak
    if len(ks) < 2:
        return float("nan")
    k_peak = ks[np.argmax(pw)]
    if k_peak < 0.5:
        return float("nan")
    # Convert grid-wavenumber to wavelength in domain units
    n_grid = max(v_field.shape)
    lambda_grid = n_grid / k_peak           # wavelength in grid cells
    lambda_domain = lambda_grid * domain_size / n_grid
    return float(lambda_domain)


# ---------------------------------------------------------------------------
# Validator entry point (returns Result-compatible dict)
# ---------------------------------------------------------------------------

def validate(verbose: bool = False) -> dict:
    """Run the Turing wavelength validation.

    Returns a dict with keys:
        pid, name, theory, predicted, observed, uncertainty,
        unit, sigma, rel_err, satisfies, status, skipped, skip_reason
    """
    result_base = dict(
        pid="PW1.1",
        name="Gray-Scott Turing pattern wavelength",
        theory="Morphogenetic Field / Meta-Boundary Dynamics",
        unit="domain units",
        status="CONSISTENT",
        satisfies=None,
    )

    def _skip(reason: str, predicted: float = float("nan")) -> dict:
        return {**result_base, "skipped": True, "skip_reason": reason,
                "predicted": predicted, "observed": float("nan"),
                "uncertainty": float("nan"), "sigma": None, "rel_err": None}

    from ..loaders import load_well_frames, WellNotAvailable, first_array

    params = GrayScottParams()
    lambda_bpr = bpr_turing_wavelength(params)

    try:
        frames = load_well_frames("gray_scott_reaction_diffusion", n=3)
    except WellNotAvailable as exc:
        return _skip(str(exc).split("\n")[0], predicted=lambda_bpr)

    # Measure wavelength across available frames and average
    wavelengths = []
    bpr_wavelengths = []
    for frame in frames:
        # Field B is the autocatalyst (activator) in The Well's Gray-Scott
        v = first_array(frame, "B", "v", "concentration_v", "activator")
        # Use actual F, k from this file's scalars for BPR prediction
        frame_params = GrayScottParams(
            F=float(frame.get("F", params.F)),
            k=float(frame.get("k", params.k)),
            D_u=params.D_u,
            D_v=params.D_v,
            domain_size=params.domain_size,
        )
        lam_bpr = bpr_turing_wavelength(frame_params)
        lam = measure_turing_wavelength(v, domain_size=params.domain_size)
        if math.isfinite(lam) and math.isfinite(lam_bpr) and lam_bpr > 0:
            wavelengths.append(lam)
            bpr_wavelengths.append(lam_bpr)
            if verbose:
                print(f"  F={frame_params.F:.4f} k={frame_params.k:.4f}"
                      f"  λ_obs={lam:.3f}  λ_BPR={lam_bpr:.3f}")

    if not wavelengths:
        return _skip("Could not extract wavelength from loaded frames", predicted=lambda_bpr)

    lambda_obs = float(np.mean(wavelengths))
    lambda_bpr_mean = float(np.mean(bpr_wavelengths))
    lambda_std = float(np.std(wavelengths)) if len(wavelengths) > 1 else lambda_bpr_mean * 0.10

    theory_uncertainty = max(lambda_std, lambda_bpr_mean * 0.10)
    sigma = abs(lambda_obs - lambda_bpr_mean) / theory_uncertainty
    rel_err = abs(lambda_obs - lambda_bpr_mean) / abs(lambda_obs) if lambda_obs != 0 else None

    if verbose:
        print(f"  Gray-Scott frames : {len(wavelengths)}")
        print(f"  λ_obs (mean±std)  : {lambda_obs:.4f} ± {lambda_std:.4f}")
        print(f"  λ_BPR (mean)      : {lambda_bpr_mean:.4f}")
        print(f"  Relative error    : {rel_err:.1%}")

    return {**result_base,
            "skipped": False, "skip_reason": None,
            "predicted": lambda_bpr_mean, "observed": lambda_obs,
            "uncertainty": theory_uncertainty, "sigma": sigma, "rel_err": rel_err}
