"""
Post neutron star merger magnetic energy spectrum validator
===========================================================

Well dataset : ``post_neutron_star_merger``
BPR prediction: PW19.1 --- Kolmogorov/KI magnetic energy spectrum

Post-merger remnants of neutron star collisions involve highly
turbulent, magnetised plasma.  In the non-relativistic limit,
BPR's substrate topology in d=3 predicts the magnetic energy
spectrum follows Kolmogorov / Kraichnan-Iroshnikov scaling:

    E_B(k) proportional to k^{-5/3}

This is the most uncertain BPR prediction tested here because
the simulation is fully relativistic MHD, while BPR's substrate
framework is Newtonian.  Theory uncertainty is increased to +/- 0.5
to account for relativistic corrections.

Method
------
1. Load magnetic_field (or velocity) from post_neutron_star_merger.
2. If magnetic_field is available, compute |B|^2 energy field.
   Otherwise fall back to velocity kinetic energy |v|^2.
3. Squeeze to 3-D, compute radial power spectrum.
4. Fit spectral index; BPR predicts alpha ~ -5/3 +/- 0.5.

Status: CONJECTURAL (Newtonian BPR applied to relativistic regime).
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


def _compute_energy_field(frame: dict, verbose: bool = False) -> np.ndarray | None:
    """Extract a 3-D energy field from the frame.

    Prefers magnetic energy |B|^2; falls back to kinetic energy |v|^2.
    """
    # Try magnetic field first
    for field_name in ("magnetic_field", "B_field", "b_field"):
        if field_name in frame and isinstance(frame[field_name], np.ndarray):
            B = np.asarray(frame[field_name], dtype=float)
            while B.ndim > 4:
                B = B[0]
            if B.ndim == 4 and B.shape[-1] in (2, 3):
                energy = np.sum(B ** 2, axis=-1)
                if verbose:
                    print(f"  Using {field_name} -> |B|^2, shape={energy.shape}")
                return energy

    # Try density for normalization
    rho = None
    for rho_name in ("density", "rho"):
        if rho_name in frame and isinstance(frame[rho_name], np.ndarray):
            rho = np.asarray(frame[rho_name], dtype=float)
            while rho.ndim > 3:
                rho = rho[0]
            break

    # Fall back to velocity
    for field_name in ("velocity", "vel"):
        if field_name in frame and isinstance(frame[field_name], np.ndarray):
            vel = np.asarray(frame[field_name], dtype=float)
            while vel.ndim > 4:
                vel = vel[0]
            if vel.ndim == 4 and vel.shape[-1] in (2, 3):
                energy = np.sum(vel ** 2, axis=-1)
                if rho is not None and rho.shape == energy.shape:
                    energy = 0.5 * rho * energy
                if verbose:
                    print(f"  Using {field_name} -> KE, shape={energy.shape}")
                return energy
            elif vel.ndim == 3:
                if verbose:
                    print(f"  Using {field_name} scalar, shape={vel.shape}")
                return vel ** 2

    return None


# ---------------------------------------------------------------------------
# Validator entry point
# ---------------------------------------------------------------------------

def validate(verbose: bool = False) -> dict:
    """Validate BPR Kolmogorov/KI prediction against post-merger NS data.

    PW19.1 -- BPR predicts magnetic/kinetic energy spectrum
    E(k) proportional to k^{-5/3} in the non-relativistic limit.
    """
    result_base = dict(
        pid="PW19.1",
        name="Post neutron star merger energy spectral index",
        theory="Substrate Topology d=3 -- Kolmogorov/KI (Newtonian limit)",
        unit="spectral index alpha",
        status="CONJECTURAL",
        satisfies=None,
    )

    from ..loaders import load_well_frames, WellNotAvailable

    bpr_alpha = -5.0 / 3.0
    theory_unc = 0.5     # increased for relativistic corrections

    def _skip(reason: str) -> dict:
        return {**result_base, "skipped": True, "skip_reason": reason,
                "predicted": bpr_alpha, "observed": float("nan"),
                "uncertainty": theory_unc, "sigma": None, "rel_err": None}

    try:
        frames = load_well_frames("post_neutron_star_merger", n=1,
                                  max_samples=1, max_timesteps=2)
    except WellNotAvailable as exc:
        return _skip(str(exc).split("\n")[0])

    spectra = []
    for frame in frames:
        try:
            energy = _compute_energy_field(frame, verbose=verbose)
            if energy is None:
                # Last resort: use first_array
                from ..loaders import first_array
                arr = first_array(frame)
                while arr.ndim > 3:
                    arr = arr[0]
                if arr.ndim == 3:
                    energy = arr ** 2
                else:
                    if verbose:
                        print(f"  Cannot extract 3-D field, shape={arr.shape}")
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
        return _skip("Could not compute spectral index from merger fields")

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
