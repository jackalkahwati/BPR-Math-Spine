"""
Spectral analysis + statistical tests.

For each (instance, phase-field-name, control-condition) we compute:
  - FFT spectrum |F|
  - top-K peaks (frequency, amplitude)
  - spectral entropy
  - peak sharpness
  - mutual information I(k ; spectral_features)  (across instances)
  - p-value for: does k lie in the FFT-predicted top-K bin set?
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


def fft_features(phi: np.ndarray, top_k: int = 5) -> dict:
    """Compute spectral features of a complex 1D field."""
    F = np.fft.fft(phi)
    A = np.abs(F)
    # Discard DC bin from peak hunt (it's usually dominant)
    A_nodc = A.copy()
    A_nodc[0] = 0.0
    if A.sum() == 0:
        return {
            "amp": A, "phase": np.angle(F),
            "top_idx": np.zeros(top_k, dtype=int),
            "top_amp": np.zeros(top_k),
            "spectral_entropy": 0.0,
            "peak_sharpness": 0.0,
        }
    P = A_nodc**2
    P /= P.sum() if P.sum() > 0 else 1.0
    H = float(-np.sum(P[P > 0] * np.log(P[P > 0])))
    top = np.argsort(-A_nodc)[:top_k]
    peak = float(A_nodc.max())
    mean = float(A_nodc.mean())
    sharpness = peak / (mean + 1e-12)
    return {
        "amp": A,
        "phase": np.angle(F),
        "top_idx": top,
        "top_amp": A_nodc[top],
        "spectral_entropy": H,
        "peak_sharpness": sharpness,
    }


def predicted_k_bins(features: dict, m: int, top_k: int = 5) -> set[int]:
    """Map FFT peak indices back to candidate W positions in [0, m).
    A peak at frequency f corresponds to a candidate period m/f; we
    return the dominant peak indices themselves as 'where the spectrum
    points'."""
    return set(int(i) for i in features["top_idx"][:top_k])


def hits_target(features: dict, k_in_window: int, top_k: int = 5,
                m: int | None = None) -> bool:
    """Did the spectrum's top-K peaks include a bin matching k?
    We count a hit if any top-K peak frequency f has k mod (m/gcd) close
    to a multiple of m/f. Simplest test: index match to k mod m."""
    if k_in_window < 0:
        return False
    return k_in_window in predicted_k_bins(features, m or 0, top_k)


def shuffle_phi(phi: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = phi.copy()
    rng.shuffle(out)
    return out


def random_phase_control(m: int, seed: int) -> np.ndarray:
    """Random unit-modulus complex field of length m."""
    rng = np.random.default_rng(seed)
    return np.exp(2j * np.pi * rng.random(m))


def mutual_information_continuous(x: np.ndarray, y: np.ndarray,
                                   bins: int = 16) -> float:
    """Histogram-based MI estimate. Both x, y are 1D real arrays of equal length."""
    if len(x) < 4:
        return 0.0
    hxy, _, _ = np.histogram2d(x, y, bins=bins)
    hx, _ = np.histogram(x, bins=bins)
    hy, _ = np.histogram(y, bins=bins)
    pxy = hxy / hxy.sum() if hxy.sum() > 0 else hxy
    px = hx / hx.sum() if hx.sum() > 0 else hx
    py = hy / hy.sum() if hy.sum() > 0 else hy
    nz = pxy > 0
    mi = 0.0
    px_full = np.outer(px, py)
    pxy_flat = pxy[nz]
    px_full_flat = px_full[nz]
    mask = px_full_flat > 0
    mi = float(np.sum(pxy_flat[mask] * np.log(pxy_flat[mask] / px_full_flat[mask])))
    return max(0.0, mi)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return boolean mask of p-values that survive BH-FDR at level alpha."""
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    threshold = alpha * ranks / n
    significant = pvals <= threshold
    # Step-up: once a p-value passes, all smaller p-values pass
    if significant.any():
        cutoff = np.max(np.where(pvals[order] <= alpha * np.arange(1, n + 1) / n,
                                  pvals[order], -np.inf))
        return pvals <= cutoff
    return np.zeros(n, dtype=bool)


@dataclass
class TrialResult:
    bits: int
    seed: int
    field_name: str
    control: str
    n: int
    k: int
    window_start: int
    window_size: int
    k_in_window: int
    top_idx: list
    top_amp: list
    spectral_entropy: float
    peak_sharpness: float
    field_cost_ops: int
    rho_baseline_ops: int
    bsgs_baseline_ops: int
    hit_topk: bool
