"""
Candidate BPR boundary phase field constructions over candidate scalars W.

All constructions take (E, G, Q, n, window_start, window_size, bpr_params).
None of them reads the secret k. Field values are returned as a complex
numpy array of length window_size, plus a group-op cost.

Cost model: we evaluate the field by walking R = (window_start)*G then
incrementing R += G. That is 1 group op per W. The initial scalar_mult
costs ~log2(window_start) group ops (counted separately).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np

from curve import Curve, Point, add, scalar_mult


@dataclass(frozen=True)
class BPRParams:
    """BPR substrate parameters (small primes). Default p=7 from
    bpr/complexity.py. The 'multi-prime' variant uses several p_i."""
    p_sub: int = 7
    multi_primes: tuple[int, ...] = (3, 5, 7, 11)


def _walk_window(E: Curve, G: Point, w0: int, m: int) -> list[Point]:
    """Return [w0*G, (w0+1)*G, ..., (w0+m-1)*G]. Uses m-1 add ops + 1 init."""
    pts = []
    R = scalar_mult(w0, G, E)
    pts.append(R)
    for _ in range(m - 1):
        R = add(R, G, E)
        pts.append(R)
    return pts


def _xcoord(P: Point, p: int) -> int:
    """Coordinate to use for fields. Use p as canonical "infinity" sentinel."""
    return p if P.is_inf else P.x


# ---------------------------------------------------------------------------
# Family A: BPR-only fields (do not touch the curve at all)
# ---------------------------------------------------------------------------

def field_A1_bpr_substrate(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """phi(W) = exp(2 pi i * W / p_sub)
    Cost: 0 group ops. Reference: pure BPR character on Z/p_sub Z."""
    W = np.arange(w0, w0 + m)
    return np.exp(2j * np.pi * W / bpr.p_sub)


def field_A2_bpr_winding_comb(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """phi(W) = sum_{p_i} exp(2 pi i * (W mod p_i) / p_i).
    Cost: 0 group ops."""
    W = np.arange(w0, w0 + m)
    out = np.zeros(m, dtype=complex)
    for p_i in bpr.multi_primes:
        out += np.exp(2j * np.pi * (W % p_i) / p_i)
    return out


# ---------------------------------------------------------------------------
# Family B: G-only fields (no Q)
# ---------------------------------------------------------------------------

def field_B1_curve_xcoord(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """phi(W) = exp(2 pi i * x(W*G) / p_field). Cost: m-1 add ops + log w0."""
    pts = _walk_window(E, G, w0, m)
    p = E.p
    xs = np.array([_xcoord(P, p) for P in pts], dtype=float)
    return np.exp(2j * np.pi * xs / p)


def field_B2_curve_substrate(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """phi(W) = exp(2 pi i * (x(W*G) mod p_sub) / p_sub).
    BPR substrate projection of curve coordinate."""
    pts = _walk_window(E, G, w0, m)
    xs = np.array([_xcoord(P, E.p) for P in pts], dtype=int)
    return np.exp(2j * np.pi * (xs % bpr.p_sub) / bpr.p_sub)


# ---------------------------------------------------------------------------
# Family C: Q-coupled fields (the candidates that could carry k)
# ---------------------------------------------------------------------------

def field_C1_q_coupled_diff(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """phi(W) = exp(2 pi i * ((x(W*G) - x(Q)) mod p) / p).
    Q enters as a constant phase shift only. By the FFT shift theorem,
    the magnitude spectrum is unchanged from B1. Phase shift is uniform."""
    pts = _walk_window(E, G, w0, m)
    p = E.p
    xQ = _xcoord(Q, p)
    xs = np.array([_xcoord(P, p) for P in pts], dtype=int)
    return np.exp(2j * np.pi * ((xs - xQ) % p) / p)


def field_C2_q_coupled_sum(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """phi(W) = exp(2 pi i * x(W*G + Q) / p), set 0 if W*G + Q = O.
    At W = n - k, the sum is the point at infinity."""
    pts = _walk_window(E, G, w0, m)
    p = E.p
    out = np.zeros(m, dtype=complex)
    for i, P in enumerate(pts):
        S = add(P, Q, E)
        if S.is_inf:
            out[i] = 0.0
        else:
            out[i] = np.exp(2j * np.pi * S.x / p)
    return out


def field_C3_q_coupled_substrate(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """BPR-substrate-projected Q-coupled difference."""
    pts = _walk_window(E, G, w0, m)
    p = E.p
    xQ = _xcoord(Q, p)
    xs = np.array([_xcoord(P, p) for P in pts], dtype=int)
    return np.exp(2j * np.pi * ((xs - xQ) % bpr.p_sub) / bpr.p_sub)


def field_C4_q_coupled_multiprime(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """Multi-prime BPR resonance of (x(W*G) - x(Q)) mod p_i."""
    pts = _walk_window(E, G, w0, m)
    p = E.p
    xQ = _xcoord(Q, p)
    xs = np.array([_xcoord(P, p) for P in pts], dtype=int)
    diff = (xs - xQ) % p
    out = np.zeros(m, dtype=complex)
    for p_i in bpr.multi_primes:
        out += np.exp(2j * np.pi * (diff % p_i) / p_i)
    return out


def field_C5_q_coupled_bilinear(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """phi(W) = exp(2 pi i * x(W*G) * x(Q) / p^2). BPR 'impedance' coupling."""
    pts = _walk_window(E, G, w0, m)
    p = E.p
    xQ = _xcoord(Q, p)
    xs = np.array([_xcoord(P, p) for P in pts], dtype=float)
    return np.exp(2j * np.pi * xs * xQ / (p * p))


def field_C6_winding_resonance(E, G, Q, n, w0, m, bpr: BPRParams) -> np.ndarray:
    """BPR 'winding x boundary' coupling:
    phi(W) = exp(2 pi i * (W mod p_sub) * (x(W*G) - x(Q)) / (p_sub * p))."""
    pts = _walk_window(E, G, w0, m)
    p = E.p
    xQ = _xcoord(Q, p)
    xs = np.array([_xcoord(P, p) for P in pts], dtype=float)
    W = np.arange(w0, w0 + m)
    diff = (xs - xQ)
    return np.exp(2j * np.pi * (W % bpr.p_sub) * diff / (bpr.p_sub * p))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FIELDS: dict[str, Callable] = {
    "A1_bpr_substrate":      field_A1_bpr_substrate,
    "A2_bpr_winding_comb":   field_A2_bpr_winding_comb,
    "B1_curve_xcoord":       field_B1_curve_xcoord,
    "B2_curve_substrate":    field_B2_curve_substrate,
    "C1_q_coupled_diff":     field_C1_q_coupled_diff,
    "C2_q_coupled_sum":      field_C2_q_coupled_sum,
    "C3_q_coupled_substrate": field_C3_q_coupled_substrate,
    "C4_q_coupled_multiprime": field_C4_q_coupled_multiprime,
    "C5_q_coupled_bilinear": field_C5_q_coupled_bilinear,
    "C6_winding_resonance":  field_C6_winding_resonance,
}


# Cost model: group ops to evaluate the field over a window of size m.
# Q-coupled-sum needs an extra add per W; A1/A2 need none.
def field_cost(name: str, m: int, w0: int) -> int:
    if name.startswith("A"):
        return 0
    if name == "C2_q_coupled_sum":
        # 1 add for W*G step + 1 add for (W*G)+Q per element
        return 2 * m + max(0, w0.bit_length())
    return m + max(0, w0.bit_length())
