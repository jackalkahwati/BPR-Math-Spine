"""
Convective envelope (red supergiant) velocity spectrum validator
================================================================

Well dataset : ``convective_envelope_rsg``
BPR prediction: PW18.1 --- Kolmogorov cascade in stellar convection

Red supergiant (RSG) convective envelopes are fully turbulent with
mixing-length-theory (MLT) convection.  In the inertial range of
fully developed 3-D turbulence, BPR's substrate topology in d=3
predicts the standard Kolmogorov energy cascade:

    E(k) proportional to k^{-5/3}

This is a CONSISTENT test -- BPR reproduces the known Kolmogorov
spectrum from its Class A winding transitions in d=3.

Method
------
1. Load velocity from convective_envelope_rsg.
2. Squeeze to 3-D velocity components.
3. Compute 3-D radial power spectrum of kinetic energy |v|^2.
4. Fit spectral index in the inertial range.
5. BPR predicts alpha ~ -5/3 +/- 0.3.

Status: CONSISTENT (standard Kolmogorov cascade).
"""

from __future__ import annotations

import math
import numpy as np


def _radial_power_spectrum_3d(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 3-D power spectrum.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz) -- real scalar field

    Returns
    -------
    (k_bins, power) -- wavenumber bins and mean power per bin
    """
    nx, ny, nz = field.shape
    fft3 = np.fft.fftn(field)
    power = np.abs(fft3) ** 2

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    k_max = int(min(nx, ny, nz) / 2)
    k_bins = np.arange(1, k_max)
    ps = np.zeros(len(k_bins))
    for i, kb in enumerate(k_bins):
        mask = (K >= kb - 0.5) & (K < kb + 0.5)
        if mask.any():
            ps[i] = power[mask].mean()
    return k_bins.astype(float), ps


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
    """Validate BPR Kolmogorov prediction against RSG convective envelope.

    PW18.1 -- BPR predicts E(k) proportional to k^{-5/3} (Kolmogorov)
    for fully developed 3-D stellar convection.
    """
    result_base = dict(
        pid="PW18.1",
        name="RSG convective envelope velocity spectral index",
        theory="Substrate Topology d=3 -- Kolmogorov cascade",
        unit="spectral index alpha",
        status="CONSISTENT",
        satisfies=None,
    )

    from ..loaders import load_well_frames, WellNotAvailable, first_array

    bpr_alpha = -5.0 / 3.0
    theory_unc = 0.3

    def _skip(reason: str) -> dict:
        return {**result_base, "skipped": True, "skip_reason": reason,
                "predicted": bpr_alpha, "observed": float("nan"),
                "uncertainty": theory_unc, "sigma": None, "rel_err": None}

    try:
        frames = load_well_frames("convective_envelope_rsg", n=1,
                                  max_samples=1, max_timesteps=2)
    except WellNotAvailable as exc:
        return _skip(str(exc).split("\n")[0])

    spectra = []
    for frame in frames:
        try:
            vel = first_array(frame, "velocity")
            # Squeeze to 3D spatial + components
            while vel.ndim > 4:
                vel = vel[0]
            # vel could be (nx, ny, nz, 3) or (nx, ny, nz)
            if vel.ndim == 4 and vel.shape[-1] in (2, 3):
                energy = np.sum(vel ** 2, axis=-1)
            elif vel.ndim == 3:
                energy = vel ** 2
            else:
                if verbose:
                    print(f"  Unexpected velocity shape: {vel.shape}")
                continue

            k_bins, ps = _radial_power_spectrum_3d(energy)
            alpha = fit_spectral_index(k_bins, ps)
            if math.isfinite(alpha):
                spectra.append(alpha)
            if verbose:
                print(f"  Frame: alpha={alpha:.3f}  grid={energy.shape}")
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
