"""
MHD turbulence energy spectrum validator
==========================================

Well dataset : ``MHD_64`` or ``MHD_256``
BPR prediction: P7.1 proxy — MHD turbulence energy spectrum power law

BPR's substrate topology predicts that collective field energy distributes
across spatial scales following E(k) ∝ k^α where α is set by the substrate
winding structure.  For 3-D MHD turbulence, BPR's Class A winding transitions
predict α ≈ −5/3 (Kolmogorov / Kraichnan–Iroshnikov spectrum).

This is a CONSISTENT test — BPR reproduces the known MHD spectrum; The Well
lets us verify the spectrum directly from simulation.

Separately: the directional Alfvén speed anisotropy is measured and reported
as a diagnostic, but noted as physically expected (driven MHD IS anisotropic
along B₀).  BPR's P7.1 isotropy claim is about GW propagation in vacuum,
not MHD turbulence driven by an external field — so that comparison was
ill-posed in the original design.

Method
------
1. Load magnetic_field and density from MHD_64.
2. Compute total magnetic energy E_B(x,y,z) = |B|²/(2μ₀ρ).
3. 3-D FFT → radial power spectrum E(k).
4. Fit log E vs log k for k in inertial range → spectral index α.
5. BPR predicts α ≈ −5/3;  observed for 3-D MHD: −5/3 to −3/2.
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def _radial_power_spectrum_3d(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 3-D power spectrum.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz) — real scalar field

    Returns
    -------
    (k_bins, power) — wavenumber bins and mean power per bin
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
    """Fit E(k) ∝ k^α in the inertial range.

    Uses the middle fraction of wavenumbers (avoids forcing scale and
    dissipation range).
    """
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
# Alfvén speed diagnostic (reported but not tested against BPR bound)
# ---------------------------------------------------------------------------

def _alfven_anisotropy(B: np.ndarray, rho: np.ndarray) -> float:
    """Directional Alfvén speed anisotropy δv/v.
    B shape (nx,ny,nz,3), rho shape (nx,ny,nz).
    """
    rho = np.where(rho > 0, rho, 1e-10)
    speeds = []
    for c in range(3):
        v = float(np.sqrt(np.mean(B[..., c] ** 2 / rho)))
        speeds.append(v)
    sa = np.array(speeds)
    return float(np.std(sa) / (np.mean(sa) + 1e-30))


# ---------------------------------------------------------------------------
# Validator entry point
# ---------------------------------------------------------------------------

def validate(verbose: bool = False) -> dict:
    """Validate MHD turbulence energy spectral index against BPR prediction.

    PW5.1 — BPR predicts spectral index α ≈ −5/3 for 3-D MHD turbulence.
    Also reports Alfvén speed anisotropy as a diagnostic (expected to be
    large for driven MHD — not a BPR test).
    """
    result_base = dict(
        pid="PW5.1",
        name="MHD turbulence energy spectral index (E∝k^α)",
        theory="Gravitational Wave / Substrate Topology (P7.1 proxy)",
        unit="spectral index α",
        status="CONSISTENT",
        satisfies=None,
    )

    from ..loaders import load_well_frames, WellNotAvailable

    bpr_alpha = -5.0 / 3.0      # Kolmogorov / KI prediction
    theory_unc = 0.15            # ±0.15 covers −5/3 to −3/2 range

    def _skip(reason: str) -> dict:
        return {**result_base, "skipped": True, "skip_reason": reason,
                "predicted": bpr_alpha, "observed": float("nan"),
                "uncertainty": theory_unc, "sigma": None, "rel_err": None}

    ds_name, grid = "MHD_64", 64
    try:
        frames = load_well_frames(ds_name, n=3)
    except WellNotAvailable:
        try:
            ds_name, grid = "MHD_256", 256
            frames = load_well_frames(ds_name, n=2)
        except WellNotAvailable as exc:
            return _skip(str(exc).split("\n")[0])

    spectra = []
    for frame in frames:
        try:
            B = np.asarray(frame["magnetic_field"], dtype=float)
            rho = np.asarray(frame["density"], dtype=float)
            # Squeeze sample dim, take last timestep
            while B.ndim > 4:
                B = B[0]
            while rho.ndim > 3:
                rho = rho[0]
            # Energy density field: |B|² / (2 ρ)
            E_field = np.sum(B ** 2, axis=-1) / (2.0 * np.where(rho > 0, rho, 1e-10))
            k_bins, ps = _radial_power_spectrum_3d(E_field)
            alpha = fit_spectral_index(k_bins, ps)
            if math.isfinite(alpha):
                spectra.append(alpha)
            aniso = _alfven_anisotropy(B, rho)
            if verbose:
                print(f"  {ds_name} Ma={frame.get('Ma','?'):.2f} Ms={frame.get('Ms','?'):.2f}"
                      f"  α={alpha:.3f}  Alfvén δv/v={aniso:.3f} (diagnostic)")
        except Exception as e:
            if verbose:
                print(f"  Frame error: {e}")

    if not spectra:
        return _skip("Could not compute spectral index from MHD frames")

    alpha_obs = float(np.mean(spectra))
    alpha_std = float(np.std(spectra)) if len(spectra) > 1 else theory_unc
    unc = max(alpha_std, theory_unc)
    sigma = abs(alpha_obs - bpr_alpha) / unc
    rel_err = abs(alpha_obs - bpr_alpha) / abs(bpr_alpha)

    return {**result_base,
            "skipped": False, "skip_reason": None,
            "predicted": bpr_alpha, "observed": alpha_obs,
            "uncertainty": unc, "sigma": sigma, "rel_err": rel_err}
