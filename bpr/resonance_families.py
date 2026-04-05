"""
Resonance Families: Integers, Quasi-Integers, and Structured Resonance
======================================================================

Farey mediant construction, quasi-integer stability, and
resonance algebra between integer attractors.

Key structures
--------------
    Farey mediant: (a+c)/(b+d)
    Resonance weight: W(p/q) = 1 / q^alpha
    Resonance algebra A with star operation
    Projection Pi_Z, weight metric d_W

References: Al-Kahwati (2026), BPR Families paper
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Farey sequence
# ---------------------------------------------------------------------------

def farey_sequence(n: int) -> List[Fraction]:
    """Generate the Farey sequence F_n (all fractions p/q with 0<=p/q<=1, q<=n).

    Parameters
    ----------
    n : int
        Maximum denominator.

    Returns
    -------
    F_n : list of Fraction, sorted ascending
    """
    fracs = set()
    for q in range(1, n + 1):
        for p in range(0, q + 1):
            fracs.add(Fraction(p, q))
    return sorted(fracs)


# ---------------------------------------------------------------------------
# Farey mediant
# ---------------------------------------------------------------------------

def farey_mediant(a_b: Tuple[int, int], c_d: Tuple[int, int]) -> Tuple[int, int]:
    """Compute the Farey mediant of a/b and c/d.

    mediant = (a + c) / (b + d)

    Parameters
    ----------
    a_b : (numerator, denominator)
    c_d : (numerator, denominator)

    Returns
    -------
    (a+c, b+d) : tuple of int
    """
    a, b = a_b
    c, d = c_d
    return (a + c, b + d)


# ---------------------------------------------------------------------------
# Farey (Stern-Brocot) tree
# ---------------------------------------------------------------------------

def farey_tree(depth: int) -> List[Tuple[int, int]]:
    """Build the Stern-Brocot / Farey tree to given depth.

    Starts with 0/1 and 1/1, and recursively inserts mediants.

    Parameters
    ----------
    depth : int
        Number of mediant insertion rounds.

    Returns
    -------
    nodes : list of (numerator, denominator) tuples, sorted by value
    """
    # Start with the boundary fractions
    tree = [(0, 1), (1, 1)]

    for _ in range(depth):
        new_tree = [tree[0]]
        for i in range(len(tree) - 1):
            new_tree.append(farey_mediant(tree[i], tree[i + 1]))
            new_tree.append(tree[i + 1])
        tree = new_tree

    # Sort by value
    tree.sort(key=lambda pq: pq[0] / pq[1])
    return tree


# ---------------------------------------------------------------------------
# Resonance weight
# ---------------------------------------------------------------------------

def resonance_weight(p: int, q: int, alpha: float = 1.0) -> float:
    """Resonance weight for the fraction p/q.

    W(p/q) = 1 / q^alpha

    Higher-denominator fractions are weaker resonances (smaller W).

    Parameters
    ----------
    p : int
        Numerator (used for completeness; weight depends only on q).
    q : int
        Denominator (must be >= 1).
    alpha : float
        Decay exponent.  alpha=1 gives harmonic weighting, alpha=2 gives
        faster suppression of high-order resonances.

    Returns
    -------
    W : float
    """
    if q < 1:
        raise ValueError(f"Denominator must be >= 1, got {q}")
    return 1.0 / q**alpha


# ---------------------------------------------------------------------------
# Quasi-integer test
# ---------------------------------------------------------------------------

def quasi_integer(x: float, resolution: float = 0.05) -> bool:
    """Test whether x is a quasi-integer at the given resolution.

    A value is quasi-integer if its distance to the nearest integer is
    less than `resolution`.

    Parameters
    ----------
    x : float
    resolution : float
        Tolerance around integers.

    Returns
    -------
    is_quasi_int : bool
    """
    return abs(x - round(x)) < resolution


# ---------------------------------------------------------------------------
# Resonance algebra: star product
# ---------------------------------------------------------------------------

def resonance_algebra_product(
    x: Tuple[int, int],
    y: Tuple[int, int],
) -> Tuple[int, int]:
    """Resonance algebra star operation (mediant-based combination).

    x star y = mediant(x, y) = (a+c, b+d)

    This defines a non-associative, commutative binary operation on
    the set of fractions that generates the Stern-Brocot tree.

    Parameters
    ----------
    x : (p, q) fraction
    y : (r, s) fraction

    Returns
    -------
    (p+r, q+s) : mediant fraction
    """
    return farey_mediant(x, y)


# ---------------------------------------------------------------------------
# Integer projection
# ---------------------------------------------------------------------------

def integer_projection(x: float) -> int:
    """Projection Pi_Z: map to nearest integer attractor.

    Parameters
    ----------
    x : float

    Returns
    -------
    nearest integer : int
    """
    return round(x)


# ---------------------------------------------------------------------------
# Weight metric
# ---------------------------------------------------------------------------

def weight_metric(
    x: Tuple[int, int],
    y: Tuple[int, int],
    alpha: float = 1.0,
) -> float:
    """Weighted distance in the resonance algebra.

    d_W(x, y) = |x_val - y_val| * (W(x) + W(y)) / 2

    where x_val = p/q and W(x) = resonance_weight(p, q, alpha).
    This metric penalises distance less between high-denominator
    (weakly resonant) fractions.

    Parameters
    ----------
    x, y : (numerator, denominator) tuples
    alpha : float
        Weight exponent.

    Returns
    -------
    d : float
    """
    x_val = x[0] / x[1]
    y_val = y[0] / y[1]
    w_x = resonance_weight(x[0], x[1], alpha)
    w_y = resonance_weight(y[0], y[1], alpha)
    return abs(x_val - y_val) * (w_x + w_y) / 2.0


# ---------------------------------------------------------------------------
# Resonance family spectrum
# ---------------------------------------------------------------------------

def resonance_family_spectrum(
    n_max: int,
    alpha: float = 1.0,
) -> List[Tuple[float, float]]:
    """Enumerate all resonance families up to denominator n_max.

    Returns (value, weight) pairs for every reduced fraction p/q
    with 0 < p/q < 1 and q <= n_max, sorted by weight (strongest first).

    Parameters
    ----------
    n_max : int
        Maximum denominator.
    alpha : float
        Weight exponent.

    Returns
    -------
    spectrum : list of (value, weight) tuples
    """
    seen = set()
    spectrum = []
    for q in range(1, n_max + 1):
        for p in range(1, q):
            f = Fraction(p, q)
            if f not in seen:
                seen.add(f)
                val = float(f)
                w = resonance_weight(f.numerator, f.denominator, alpha)
                spectrum.append((val, w))
    # Sort by weight descending (strongest resonances first)
    spectrum.sort(key=lambda vw: -vw[1])
    return spectrum
