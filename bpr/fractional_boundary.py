"""
Fractional Boundary Resonance Index
====================================

Extends BPR to fractal boundaries with fractional dimension D_S.
Derives transport scaling laws on Sierpinski-type geometries.

Key equations
-------------
    BRI(E) = sum_n sgn(v_n) chi_n(E)
    Q(L) ~ L^{gamma(D_S)}       for 1 < D_S < 2
    G ~ L^{D_S - 1}             conductance scaling
    ell ~ xi^{2 - D_S}          localisation length

References: Al-Kahwati (2026), 158D upgrade paper
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Boundary Resonance Index
# ---------------------------------------------------------------------------

def boundary_resonance_index(
    velocities: np.ndarray,
    chi_values: np.ndarray,
    E: float,
) -> float:
    """Compute the Boundary Resonance Index at energy E.

    BRI(E) = sum_n sgn(v_n) * chi_n(E)

    Parameters
    ----------
    velocities : 1-D array
        Group velocities v_n at energy E for each channel n.
    chi_values : 1-D array
        Characteristic function (indicator / weight) chi_n(E) for each channel.
    E : float
        Energy parameter (used for documentation; chi_values should already
        be evaluated at E).

    Returns
    -------
    bri : float
    """
    signs = np.sign(velocities)
    return float(np.sum(signs * chi_values))


# ---------------------------------------------------------------------------
# Transport scaling on fractal boundaries
# ---------------------------------------------------------------------------

def transport_scaling(L: np.ndarray, D_S: float) -> np.ndarray:
    """Conductance scaling on a fractal boundary.

    G(L) ~ L^{D_S - 1}

    Parameters
    ----------
    L : array-like
        System sizes.
    D_S : float
        Fractal (Hausdorff) dimension of the boundary, 1 < D_S < 2.

    Returns
    -------
    G : ndarray, same shape as L
    """
    L = np.asarray(L, dtype=float)
    return L ** (D_S - 1.0)


def quality_factor_scaling(
    L: np.ndarray,
    D_S: float,
    gamma_func: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """Resonance quality factor scaling Q(L) ~ L^{gamma(D_S)}.

    Parameters
    ----------
    L : array-like
        System sizes.
    D_S : float
        Fractal dimension.
    gamma_func : callable, optional
        Maps D_S -> exponent gamma.  Default: gamma(D_S) = 2*D_S - 2
        (quadratic enhancement for fractal boundaries).

    Returns
    -------
    Q : ndarray
    """
    L = np.asarray(L, dtype=float)
    if gamma_func is None:
        gamma = 2.0 * D_S - 2.0
    else:
        gamma = gamma_func(D_S)
    return L ** gamma


def localization_length(xi: np.ndarray, D_S: float) -> np.ndarray:
    """Effective localisation length on a fractal boundary.

    ell ~ xi^{2 - D_S}

    Parameters
    ----------
    xi : array-like
        Bulk correlation length.
    D_S : float
        Fractal dimension.

    Returns
    -------
    ell : ndarray
    """
    xi = np.asarray(xi, dtype=float)
    return xi ** (2.0 - D_S)


# ---------------------------------------------------------------------------
# Sierpinski geometry
# ---------------------------------------------------------------------------

def sierpinski_dimension(level: int = 1) -> float:
    """Hausdorff dimension of the Sierpinski triangle.

    D_S = ln(3) / ln(2) ~ 1.585, independent of iteration level.

    Parameters
    ----------
    level : int
        Iteration level (returned for consistency; dimension is constant).

    Returns
    -------
    D_S : float
    """
    return np.log(3.0) / np.log(2.0)


# ---------------------------------------------------------------------------
# Geometry swap test
# ---------------------------------------------------------------------------

def geometry_swap_test(
    L_values: np.ndarray,
    D_S_values: np.ndarray,
) -> np.ndarray:
    """Compare conductance G(L) across different fractal dimensions.

    Parameters
    ----------
    L_values : 1-D array of system sizes
    D_S_values : 1-D array of fractal dimensions to compare

    Returns
    -------
    G_table : ndarray shape (len(D_S_values), len(L_values))
        G_table[i, j] = L_values[j]^{D_S_values[i] - 1}
    """
    L_values = np.asarray(L_values, dtype=float)
    D_S_values = np.asarray(D_S_values, dtype=float)
    # Broadcast: (n_D, 1) ** exponent with (1, n_L) -> (n_D, n_L)
    exponents = D_S_values[:, None] - 1.0
    return L_values[None, :] ** exponents


# ---------------------------------------------------------------------------
# Resonance frequency window
# ---------------------------------------------------------------------------

def frequency_window(D_S: float, L: float, c_eff: float) -> Tuple[float, float]:
    """Compute the resonance frequency window for a fractal boundary.

    The fundamental resonance occurs at f_0 = c_eff / (2L).
    The fractal boundary broadens the window by a factor related to D_S:

        Delta_f / f_0 ~ L^{-(2 - D_S)}

    Parameters
    ----------
    D_S : float
        Fractal dimension of the boundary.
    L : float
        System size.
    c_eff : float
        Effective wave speed in the medium.

    Returns
    -------
    (f_center, delta_f) : tuple of float
        Centre frequency and half-width of the resonance window.
    """
    f_center = c_eff / (2.0 * L)
    # Fractional broadening
    delta_f = f_center * L ** (-(2.0 - D_S))
    return float(f_center), float(delta_f)
