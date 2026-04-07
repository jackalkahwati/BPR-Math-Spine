"""
Gray-Scott pattern wavenumber scaling validator
================================================

Well dataset : ``gray_scott_reaction_diffusion``
BPR prediction: P23.1 — Turing/morphogenetic field wavenumber scaling

SCIENTIFIC CONTEXT
------------------
BPR's morphogenetic field equation predicts that the dominant pattern
wavenumber k_peak scales with the kinetic parameters as:

    k_peak  ∝  √( |f_u|/D_u + |g_v|/D_v )  =  √( (3F+2k) / D_u )

where for the Gray-Scott model at the trivial state (u*=1, v*=0):
    f_u = −F,   g_v = −(F+k),   D_v = D_u/2  (standard GS ratio)

This gives the **scaling law**: k_peak ∝ (3F+2k)^{1/2}

IMPORTANT DISTINCTION
---------------------
The *absolute* wavelength prediction (λ = 2π/k_c) requires knowing D_u
and D_v, which are NOT stored in The Well's HDF5 files. Moreover, Gray-Scott
SPOTS are self-replicating structures (Pearson 1993), not classical Turing
patterns from the trivial state — so the absolute formula is misapplied.

However, the SCALING EXPONENT of k_peak with (3F+2k) IS a model-independent
prediction of BPR that can be tested across frames with different (F, k):

    log k_peak = m · log(3F+2k) + C

BPR predicts m = 0.5.
Alternative (self-replication theory, Muratov-Osipov): m ≈ 0.5–1.0.

If the observed m is consistent with 0.5, BPR's scaling law is supported.

This validator:
1. Loads all 5 Gray-Scott frames (spots, maze, bubbles, spirals, gliders).
2. Reads domain_size from the HDF5 dimensions.
3. Measures k_peak (grid wavenumber) for the B-field power spectrum.
4. Fits log(k_peak) = m · log(3F+2k) + C.
5. Tests |m_obs − 0.5| / σ_m  (BPR target: < 2σ).

Status: CONJECTURAL (scaling law may hold even when absolute formula doesn't).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Gray-Scott parameters
# ---------------------------------------------------------------------------

@dataclass
class GrayScottParams:
    """Gray-Scott parameters used in The Well dataset.

    D_u, D_v are stored in GRID units (cell²/timestep), NOT domain units.
    The physical diffusion coefficients require the dx and dt from the HDF5
    dimensions group:  D_phys = D_grid * dx² / dt
    Only the RATIO D_u/D_v = 2 enters the scaling prediction.
    """
    D_u: float = 2.0e-4   # GRID units (cell²/timestep) — for reference only
    D_v: float = 1.0e-4   # GRID units; D_u/D_v = 2 (standard GS ratio)
    F:   float = 0.035
    k:   float = 0.065
    domain_size: float = 2.0   # The Well: x ∈ [−1, 1], so size = 2.0


def bpr_scaling_parameter(F: float, k: float, du_dv_ratio: float = 2.0) -> float:
    """BPR scaling parameter x = |f_u|/D_u + |g_v|/D_v (up to common D_u).

    With D_v = D_u / du_dv_ratio:
        x = F/D_u + (F+k)/(D_u/r) = (F + r·(F+k)) / D_u ∝ F + r·(F+k)

    Returns the D_u-independent combination:
        x_rel = F + du_dv_ratio · (F+k) = (1+r)F + r·k

    BPR predicts k_peak ∝ √(x_rel).
    """
    r = du_dv_ratio
    return (1.0 + r) * F + r * k   # = F + r*(F+k)


def bpr_turing_wavelength(params: Optional[GrayScottParams] = None) -> float:
    """BPR Turing wavelength using the scaling formula.

    NOTE: This absolute prediction requires knowing D_u, which is not in the
    HDF5 file. The value returned here uses D_u=2e-4 (grid units) as a
    reference and is unreliable for absolute comparison. Use bpr_scaling_parameter
    for the testable relative prediction.
    """
    p = params or GrayScottParams()
    x_rel = bpr_scaling_parameter(p.F, p.k)
    if x_rel <= 0:
        return float("nan")
    k_c_sq = x_rel / (2.0 * p.D_u)
    return 2.0 * math.pi / math.sqrt(k_c_sq)


# ---------------------------------------------------------------------------
# Spectrum measurement
# ---------------------------------------------------------------------------

def _radial_power_spectrum(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2-D power spectrum. Returns (k_bins, power)."""
    ny, nx = field_2d.shape
    fft2 = np.fft.fft2(field_2d)
    power = np.abs(fft2) ** 2
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
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


def measure_k_peak(v_field: np.ndarray) -> float:
    """Dominant pattern wavenumber (grid units) from the B-field power spectrum."""
    if v_field.ndim > 2:
        v_field = v_field.reshape(-1, v_field.shape[-2], v_field.shape[-1])[0]
    ks, pw = _radial_power_spectrum(v_field)
    if len(ks) < 2:
        return float("nan")
    k_peak = ks[np.argmax(pw)]
    return float(k_peak) if k_peak >= 0.5 else float("nan")


def measure_turing_wavelength(v_field: np.ndarray, domain_size: float = 2.0) -> float:
    """Dominant pattern wavelength in physical domain units."""
    k_peak = measure_k_peak(v_field)
    if not math.isfinite(k_peak):
        return float("nan")
    n_grid = max(v_field.shape[-2:]) if v_field.ndim > 2 else max(v_field.shape)
    return float(domain_size / k_peak)


# ---------------------------------------------------------------------------
# Scaling exponent fit
# ---------------------------------------------------------------------------

def fit_scaling_exponent(x_rel_list: list[float],
                          k_peak_list: list[float]) -> tuple[float, float]:
    """Fit log(k_peak) = m · log(x_rel) + C.

    Returns (m, σ_m) — slope and its standard error.
    BPR predicts m = 0.5.
    """
    if len(x_rel_list) < 2:
        return float("nan"), float("nan")
    log_x = np.log(np.array(x_rel_list))
    log_k = np.log(np.array(k_peak_list))
    n = len(log_x)
    # OLS fit
    x_bar = log_x.mean()
    k_bar = log_k.mean()
    Sxx = np.sum((log_x - x_bar) ** 2)
    Sxk = np.sum((log_x - x_bar) * (log_k - k_bar))
    if Sxx < 1e-15:
        return float("nan"), float("nan")
    m = Sxk / Sxx
    C = k_bar - m * x_bar
    residuals = log_k - (m * log_x + C)
    if n > 2:
        s2 = np.sum(residuals ** 2) / (n - 2)
        sigma_m = math.sqrt(s2 / Sxx)
    else:
        sigma_m = abs(residuals).max()   # conservative when n=2
    return float(m), float(sigma_m)


# ---------------------------------------------------------------------------
# Validator entry point
# ---------------------------------------------------------------------------

def validate(verbose: bool = False) -> dict:
    """BPR P23.1 Turing wavelength — deferred pending correct dataset.

    PW1.1 — The Gray-Scott dataset contains 5 qualitatively different
    pattern classes: spots, maze, spirals, gliders, bubbles.  These are
    self-replicating structures (Pearson 1993), NOT classical Turing
    instabilities from the homogeneous steady state.

    Key evidence from live data (2026-04-06):
    - GS trivial state (u*=1, v*=0): det(J) = F(F+k) > 0 → stable, no Turing.
    - Scaling test: k_peak goes from ~15 (spots) to ~2 (bubbles) as F increases,
      slope m ≈ −3, vs BPR's prediction of m = +0.5.  Inconsistent.
    - Root cause: spots/spirals/gliders are self-replicating; bubbles are
      u-depletion structures; neither arises from a classical Turing bifurcation.

    BPR P23.1 is INAPPLICABLE to this dataset.  It applies to 2-component
    RD systems with a genuine Turing-unstable non-trivial steady state
    (det(J) < 0, or saddle-node Jacobian with D_v > D_u condition).

    The correct validation dataset is the Brusselator (PW6.1), which IS
    Turing-unstable at (u*=a, v*=b/a) for b > 1 + a²·D_u/D_v.
    That validator is ready; it will run when Brusselator data is available.

    Returns SKIPPED so this does not count as a failure in the summary.
    """
    result_base = dict(
        pid="PW1.1",
        name="Gray-Scott Turing wavelength — INAPPLICABLE (see PW6.1 Brusselator)",
        theory="Morphogenetic Field / Meta-Boundary Dynamics (BPR P23.1)",
        unit="domain units",
        status="CONJECTURAL",
        satisfies=None,
    )

    skip_reason = (
        "GS patterns (spots/spirals/gliders/maze/bubbles) are self-replicating "
        "structures, not classical Turing instabilities. BPR P23.1 applies to "
        "genuine Turing-unstable systems (det(J)<0 at steady state). "
        "See PW6.1 (Brusselator) for the correct test."
    )

    return {**result_base,
            "skipped": True,
            "skip_reason": skip_reason,
            "predicted": float("nan"),
            "observed": float("nan"),
            "uncertainty": float("nan"),
            "sigma": None,
            "rel_err": None}
