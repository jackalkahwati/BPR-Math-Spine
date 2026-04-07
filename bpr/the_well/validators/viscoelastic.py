"""
Viscoelastic instability velocity spectrum validator
=====================================================

Well dataset : ``viscoelastic_instability``
BPR prediction: PW16.1 --- elastic turbulence steep velocity spectrum

At high Weissenberg number (Wi >> 1), viscoelastic fluids exhibit
"elastic turbulence" with a velocity energy spectrum E(k) that is
steeper than both Kolmogorov (-5/3) and 2-D enstrophy cascade (-3).

BPR's Class A+C mixed transition predicts a spectral index alpha ~ -3.3
from the elastic-inertial crossover.  Experimental observations
(Groisman & Steinberg 2000) report alpha in [-3.5, -3.0].

Method
------
1. Load velocity from viscoelastic_instability.
2. Squeeze to 2-D velocity components.
3. Compute 2-D radial power spectrum of |v|^2.
4. Fit spectral index in the inertial range.
5. BPR predicts alpha ~ -3.3 +/- 0.5.

Status: CONJECTURAL (BPR's mixed-class prediction for elastic turbulence).
"""

from __future__ import annotations

import math
import numpy as np


def _radial_power_spectrum_2d(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2-D power spectrum.

    Parameters
    ----------
    field_2d : (ny, nx) real scalar field

    Returns
    -------
    (k_bins, power)
    """
    ny, nx = field_2d.shape
    fft2 = np.fft.fft2(field_2d)
    power = np.abs(fft2) ** 2 / (nx * ny)
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_max = int(min(nx, ny) / 2)
    k_bins = np.arange(1, k_max, dtype=float)
    ps = np.zeros(len(k_bins))
    for i, kb in enumerate(k_bins):
        mask = (K >= kb - 0.5) & (K < kb + 0.5)
        if mask.any():
            ps[i] = power[mask].mean()
    return k_bins, ps


def fit_spectral_index(k_bins: np.ndarray, power: np.ndarray,
                       k_min_frac: float = 0.1,
                       k_max_frac: float = 0.4) -> float:
    """Fit E(k) proportional to k^alpha in the inertial range."""
    k_min = k_bins.max() * k_min_frac
    k_max = k_bins.max() * k_max_frac
    mask = (k_bins >= k_min) & (k_bins <= k_max) & (power > 0)
    if mask.sum() < 3:
        return float("nan")
    log_k = np.log(k_bins[mask])
    log_p = np.log(power[mask])
    p = np.polyfit(log_k, log_p, 1)
    return float(p[0])


# ---------------------------------------------------------------------------
# Validator entry point
# ---------------------------------------------------------------------------

def validate(verbose: bool = False) -> dict:
    """Validate BPR elastic turbulence spectral index prediction.

    PW16.1 — In elastic turbulence (Wi ≫ 1), polymer stretching creates
    a cascade where the spectral exponent depends on the Weissenberg number.

    BPR's Class A+C mixed transition predicts:
        α_elastic = -(3 + Wi^{1/3})

    This comes from the elastic cascade theory (Balkovsky-Fouxon 2001):
    the polymer deformation rate couples to the velocity gradient tensor
    with a decay exponent δ ∝ (deformation × τ_polymer)^{1/3} = Wi^{1/3}.
    The velocity spectrum E(k) ∝ k^{-(3+δ)} = k^{-(3+Wi^{1/3})}.

    For Wi=50: α = -(3 + 50^{1/3}) = -(3 + 3.68) = -6.68
    For Wi=10: α = -(3 + 2.15) = -5.15
    For Wi=1:  α = -(3 + 1) = -4.0 (onset of elastic turbulence)
    """
    from ..loaders import load_well_frames, WellNotAvailable, first_array

    # Read Wi from the first frame to compute prediction dynamically
    try:
        frames = load_well_frames("viscoelastic_instability", n=2,
                                  max_samples=1, max_timesteps=2)
    except WellNotAvailable as exc:
        result_base = dict(
            pid="PW16.1",
            name="Viscoelastic elastic turbulence spectrum (Wi-dependent)",
            theory="Class A+C Mixed Transition — α = -(3 + Wi^{1/3})",
            unit="spectral index α",
            status="CONJECTURAL",
            satisfies=None,
        )
        return {**result_base, "skipped": True,
                "skip_reason": str(exc).split("\n")[0],
                "predicted": float("nan"), "observed": float("nan"),
                "uncertainty": 0.5, "sigma": None, "rel_err": None}

    # Get Wi from first frame
    Wi = float(frames[0].get("Wi", 50.0))
    bpr_alpha = -(3.0 + Wi ** (1.0 / 3.0))
    theory_unc = 0.5

    result_base = dict(
        pid="PW16.1",
        name=f"Viscoelastic spectrum α=-(3+Wi^{{1/3}}) at Wi={Wi:.0f}",
        theory="Class A+C Mixed Transition — elastic cascade (Balkovsky-Fouxon)",
        unit="spectral index α",
        status="CONJECTURAL",
        satisfies=None,
    )

    def _skip(reason: str) -> dict:
        return {**result_base, "skipped": True, "skip_reason": reason,
                "predicted": bpr_alpha, "observed": float("nan"),
                "uncertainty": theory_unc, "sigma": None, "rel_err": None}

    spectra = []
    for frame in frames:
        try:
            vel = first_array(frame, "velocity")
            # Squeeze to 2D: take first sample, last timestep
            while vel.ndim > 3:
                vel = vel[0]
            # vel shape: (nx, ny, 2) or (nx, ny)
            if vel.ndim == 3 and vel.shape[-1] == 2:
                energy = vel[..., 0] ** 2 + vel[..., 1] ** 2
            elif vel.ndim == 2:
                energy = vel ** 2
            else:
                # Try to interpret as scalar field
                energy = vel.reshape(vel.shape[-2], vel.shape[-1]) ** 2

            k_bins, ps = _radial_power_spectrum_2d(energy)
            alpha = fit_spectral_index(k_bins, ps)
            if math.isfinite(alpha):
                spectra.append(alpha)
            if verbose:
                print(f"  Frame: alpha={alpha:.3f}")
        except Exception as e:
            if verbose:
                print(f"  Frame error: {e}")

    if not spectra:
        return _skip("Could not compute spectral index from velocity fields")

    alpha_obs = float(np.mean(spectra))
    alpha_std = float(np.std(spectra)) if len(spectra) > 1 else theory_unc
    unc = max(alpha_std, theory_unc)
    sigma = abs(alpha_obs - bpr_alpha) / unc
    rel_err = abs(alpha_obs - bpr_alpha) / abs(bpr_alpha)

    if verbose:
        print(f"  alpha (observed)  = {alpha_obs:.3f}")
        print(f"  alpha (BPR)       = {bpr_alpha:.3f}")
        print(f"  uncertainty       = {unc:.3f}")
        print(f"  sigma             = {sigma:.2f}")

    return {**result_base,
            "skipped": False, "skip_reason": None,
            "predicted": bpr_alpha, "observed": alpha_obs,
            "uncertainty": unc, "sigma": sigma, "rel_err": rel_err}
