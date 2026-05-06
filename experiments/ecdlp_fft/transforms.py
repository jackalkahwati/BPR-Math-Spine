"""
Transform library for extended BPR-ECDLP analysis.

Each function takes a complex 1-D field phi(W) on [w0, w0+m) plus optional
auxiliary inputs (E, G, Q, n, w0, BPR params) and returns a feature dict.

We separate transforms into families to make per-family verdicts clean.

Cost model (group ops): all transforms below operate on a precomputed phi
array, so they are post-processing -- 0 group ops on top of the phi build.
The exception is the multiplicative character sum, which can be evaluated
directly on x(W*G) coords (already produced when phi was built).
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# Family 1: Phase-sensitive Fourier
# ---------------------------------------------------------------------------

def phase_features(phi: np.ndarray) -> dict:
    """Phase-sensitive features of the FFT, beyond magnitude."""
    F = np.fft.fft(phi)
    A = np.abs(F)
    phase = np.angle(F)
    A_nodc = A.copy()
    A_nodc[0] = 0.0
    if A_nodc.sum() == 0:
        return dict(phase_mean=0.0, phase_circvar=0.0, phase_slope=0.0,
                    phase_coherence=0.0, phase_top_arg=0.0)
    # Circular mean and variance of *amplitude-weighted* phases.
    w = A_nodc / A_nodc.sum()
    z = (w * np.exp(1j * phase)).sum()
    phase_mean = float(np.angle(z))
    phase_circvar = float(1.0 - np.abs(z))
    # Linear phase slope: regression of unwrapped phase vs frequency index in top-K bins
    top = np.argsort(-A_nodc)[: max(8, len(A) // 32)]
    if len(top) >= 2:
        x = top.astype(float)
        y = np.unwrap(phase[top])
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0
    # Phase coherence between consecutive bins (mean cos of phase diff)
    pdiff = np.diff(phase)
    phase_coherence = float(np.mean(np.cos(pdiff)))
    # Phase at top peak
    top_peak = int(np.argmax(A_nodc))
    phase_top_arg = float(phase[top_peak])
    return dict(phase_mean=phase_mean, phase_circvar=phase_circvar,
                phase_slope=slope, phase_coherence=phase_coherence,
                phase_top_arg=phase_top_arg)


def autocorr_features(phi: np.ndarray, max_lag: int = 16) -> dict:
    """Autocorrelation of phi at small lags. ACF computed via FFT."""
    m = len(phi)
    centered = phi - phi.mean()
    F = np.fft.fft(centered, n=2 * m)
    acf = np.fft.ifft(F * np.conj(F)).real[:m]
    if acf[0] != 0:
        acf = acf / acf[0]
    out = {}
    for lag in (1, 2, 4, 8, 16):
        if lag < m:
            out[f"acf_lag{lag}"] = float(acf[lag])
        else:
            out[f"acf_lag{lag}"] = 0.0
    out["acf_max_abs_lag"] = float(np.max(np.abs(acf[1:max_lag + 1])))
    return out


def bispectrum_feature(phi: np.ndarray, max_freq: int = 32) -> dict:
    """Bispectrum peak in a small low-freq region. Cost O(max_freq^2)."""
    F = np.fft.fft(phi)
    K = min(max_freq, len(F) // 4)
    B_max = 0.0
    B_sum = 0.0
    for f1 in range(1, K):
        for f2 in range(1, K):
            v = F[f1] * F[f2] * np.conj(F[(f1 + f2) % len(F)])
            B_sum += abs(v)
            if abs(v) > B_max:
                B_max = abs(v)
    return dict(bispectrum_max=float(B_max),
                bispectrum_mean=float(B_sum / max(1, (K - 1) ** 2)))


# ---------------------------------------------------------------------------
# Family 2: Localized spectral
# ---------------------------------------------------------------------------

def stft_features(phi: np.ndarray, win: int = 32) -> dict:
    """Sliding-window FFT energy variance across windows."""
    m = len(phi)
    if win > m:
        win = m
    n_wins = m - win + 1
    energies = np.zeros(n_wins)
    for i in range(n_wins):
        w = phi[i:i + win]
        F = np.fft.fft(w)
        A = np.abs(F)
        A[0] = 0
        energies[i] = float((A ** 2).sum())
    return dict(stft_energy_mean=float(energies.mean()),
                stft_energy_std=float(energies.std()),
                stft_energy_cv=float(energies.std() / (energies.mean() + 1e-12)))


def _ricker(M: int, w: float) -> np.ndarray:
    """Ricker (Mexican hat) wavelet, hand-rolled (scipy.signal.ricker
    was removed in scipy 1.15+). Equivalent to the historical formula."""
    A = 2 / (np.sqrt(3 * w) * np.pi ** 0.25)
    t = np.arange(M) - (M - 1) / 2.0
    x = (t / w) ** 2
    return A * (1 - x) * np.exp(-x / 2)


def wavelet_features(phi: np.ndarray, n_scales: int = 8) -> dict:
    """Continuous wavelet transform energy distribution across scales.
    Real Ricker wavelet on |phi|."""
    sig = np.abs(phi).astype(float)
    if sig.max() == sig.min():
        # constant amplitude (e.g., unit-modulus phi) -> wavelet response 0
        return dict(wavelet_max=0.0, wavelet_entropy=0.0,
                    wavelet_top_scale_idx=0)
    sig = sig - sig.mean()
    widths = np.geomspace(2, max(8, len(phi) // 4), n_scales)
    coefs = np.zeros((n_scales, len(sig)))
    for i, w in enumerate(widths):
        ker = _ricker(min(int(10 * w) + 1, len(sig)), float(w))
        coefs[i] = signal.convolve(sig, ker, mode="same")
    energies = (coefs ** 2).sum(axis=1)
    if energies.sum() > 0:
        p = energies / energies.sum()
        wav_entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    else:
        wav_entropy = 0.0
    return dict(wavelet_max=float(np.abs(coefs).max()),
                wavelet_entropy=wav_entropy,
                wavelet_top_scale_idx=int(np.argmax(energies)))


def multiscale_entropy(phi: np.ndarray, scales: tuple = (1, 2, 4, 8)) -> dict:
    """Sample entropy at multiple scales of |phi|."""
    sig = np.abs(phi)
    out = {}
    for s in scales:
        if s == 1:
            sub = sig
        else:
            n_sub = len(sig) // s
            sub = sig[: n_sub * s].reshape(n_sub, s).mean(axis=1)
        # Discretize into 8 bins, compute Shannon entropy
        if len(sub) < 4 or (sub.max() - sub.min()) < 1e-12:
            out[f"mse_s{s}"] = 0.0
            continue
        h, _ = np.histogram(sub, bins=8)
        p = h / h.sum() if h.sum() > 0 else h
        out[f"mse_s{s}"] = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return out


# ---------------------------------------------------------------------------
# Family 3: Finite-field / modular transforms
# ---------------------------------------------------------------------------

def char_sum_features(xs: np.ndarray, p: int) -> dict:
    """Multiplicative character sum (Legendre): for each x in window, compute
    Legendre symbol (x | p), then summary statistics of the running sum.

    For a uniformly-distributed sequence, the Legendre walk is a +/-1 random
    walk. Bias would indicate non-random structure.
    """
    # Filter out 0 and "infinity" sentinel (xs may contain p as inf marker)
    valid_mask = (xs != 0) & (xs < p)
    legendre = np.zeros(len(xs), dtype=int)
    for i, x in enumerate(xs):
        if not valid_mask[i]:
            legendre[i] = 0
            continue
        v = pow(int(x), (p - 1) // 2, p)
        legendre[i] = -1 if v == p - 1 else (1 if v == 1 else 0)
    cum = np.cumsum(legendre)
    return dict(legendre_mean=float(legendre.mean()),
                legendre_walk_max=float(np.max(np.abs(cum))),
                legendre_walk_end=float(cum[-1]),
                legendre_runs=int(np.sum(np.diff(legendre) != 0)))


def additive_char_sum(xs: np.ndarray, p: int, a_list=(1, 2, 3, 5, 7)) -> dict:
    """Additive character sums S_a = sum_W exp(2 pi i * a * x(W*G) / p).
    For pseudo-random x, |S_a| ~ sqrt(m) by Weil-style bounds.
    """
    m = len(xs)
    feats = {}
    for a in a_list:
        z = np.exp(2j * np.pi * a * xs.astype(float) / p)
        S = z.sum()
        feats[f"add_char_a{a}_abs"] = float(abs(S)) / np.sqrt(max(1, m))
        feats[f"add_char_a{a}_arg"] = float(np.angle(S))
    return feats


def ntt_features(seq: np.ndarray, q: int = 257) -> dict:
    """Number-theoretic transform of (seq mod q). q must be a small prime
    with q-1 a multiple of len(seq). For len 256, q=257 works."""
    m = len(seq)
    if (q - 1) % m != 0:
        # Fall back: use FFT magnitude as proxy
        F = np.fft.fft(seq)
        A = np.abs(F)
        A[0] = 0
        return dict(ntt_max=float(A.max()), ntt_entropy=0.0)
    # Primitive root of order m: g^((q-1)/m) mod q where g is a primitive root mod q
    # For q=257, 3 is a primitive root.
    g = 3
    omega = pow(g, (q - 1) // m, q)
    # NTT_k = sum_j seq[j] * omega^(j*k) mod q
    s = (seq.astype(int) % q)
    out = np.zeros(m, dtype=int)
    pwr = 1
    omega_pwrs = np.zeros(m, dtype=int)
    for i in range(m):
        omega_pwrs[i] = pwr
        pwr = (pwr * omega) % q
    for k in range(m):
        acc = 0
        idx = 0
        for j in range(m):
            acc = (acc + int(s[j]) * int(omega_pwrs[idx])) % q
            idx = (idx + k) % m
        out[k] = acc
    out_no_dc = out.copy()
    out_no_dc[0] = 0
    A = np.abs(out_no_dc.astype(float))
    return dict(ntt_max=float(A.max()), ntt_argmax=int(A.argmax()))


# ---------------------------------------------------------------------------
# Family 4: Graph / topological
# ---------------------------------------------------------------------------

def graph_laplacian_spectrum(xs: np.ndarray, k_neighbors: int = 4) -> dict:
    """Build a k-NN graph on the x-coordinate sequence as 1-D points,
    compute Laplacian spectrum. Spectral gap is a topological feature.

    For our use, treat xs as positions in F_p; nearest-neighbor in the
    natural integer ordering would just recover sequential structure.
    Instead we use nearest neighbors by *value* (x coord magnitude).
    """
    m = len(xs)
    if m < 4:
        return dict(laplacian_spectral_gap=0.0, laplacian_top_eig=0.0)
    pts = xs.astype(float)
    # Pairwise abs differences (1-D)
    diff = np.abs(pts[:, None] - pts[None, :])
    np.fill_diagonal(diff, np.inf)
    # k-NN adjacency
    A = np.zeros((m, m))
    for i in range(m):
        nn = np.argpartition(diff[i], k_neighbors)[:k_neighbors]
        A[i, nn] = 1
    A = (A + A.T) / 2  # symmetrize
    A[A > 0] = 1
    D = np.diag(A.sum(axis=1))
    L = D - A
    # Spectrum (just the smallest nonzero and largest)
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    spectral_gap = float(eigs[1]) if len(eigs) > 1 else 0.0
    return dict(laplacian_spectral_gap=spectral_gap,
                laplacian_top_eig=float(eigs[-1]))


# ---------------------------------------------------------------------------
# Master feature extractor
# ---------------------------------------------------------------------------

def all_features(phi: np.ndarray, *, xs: np.ndarray | None = None,
                 p: int | None = None) -> dict:
    """Run every transform on a single phi (and optional integer xs).

    Returns a flat dict of feature_name -> scalar.
    """
    feats = {}
    feats.update({f"ph_{k}": v for k, v in phase_features(phi).items()})
    feats.update({f"ac_{k}": v for k, v in autocorr_features(phi).items()})
    feats.update({f"bs_{k}": v for k, v in bispectrum_feature(phi).items()})
    feats.update({f"st_{k}": v for k, v in stft_features(phi).items()})
    feats.update({f"wv_{k}": v for k, v in wavelet_features(phi).items()})
    feats.update({f"mse_{k}": v for k, v in multiscale_entropy(phi).items()})
    if xs is not None and p is not None:
        feats.update({f"ch_{k}": v for k, v in char_sum_features(xs, p).items()})
        feats.update({f"ad_{k}": v for k, v in additive_char_sum(xs, p).items()})
        feats.update({f"gl_{k}": v for k, v in graph_laplacian_spectrum(xs).items()})
    return feats
