"""Axisymmetric field prototypes for Helionis geometry exploration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisymmetricFieldMap:
    """R-Z field arrays for early geometry visualization."""

    r_grid: np.ndarray
    z_grid: np.ndarray
    flux_webers: np.ndarray
    b_r_t: np.ndarray
    b_z_t: np.ndarray


def axisymmetric_frc_field_map(
    separatrix_radius_m: float,
    length_m: float,
    axial_field_t: float,
    radial_points: int = 80,
    axial_points: int = 120,
) -> AxisymmetricFieldMap:
    """Return a simple analytic FRC-like reversed-field map.

    This is a geometry visualization proxy, not an equilibrium solver. The flux
    function creates reversed axial field near the core and external guide field
    outside the separatrix so designers can start reasoning about packaging.
    """
    if separatrix_radius_m <= 0:
        raise ValueError("separatrix_radius_m must be positive")
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    if axial_field_t <= 0:
        raise ValueError("axial_field_t must be positive")
    if radial_points < 4 or axial_points < 4:
        raise ValueError("field map grids need at least 4 points per axis")

    r = np.linspace(0.0, 2.0 * separatrix_radius_m, radial_points)
    z = np.linspace(-0.5 * length_m, 0.5 * length_m, axial_points)
    z_grid, r_grid = np.meshgrid(z, r, indexing="ij")

    radial_shape = (r_grid / separatrix_radius_m) ** 2
    axial_taper = np.cos(np.pi * z_grid / length_m) ** 2
    flux = 0.5 * axial_field_t * r_grid**2 * (radial_shape - 1.0) * axial_taper

    dpsi_dz, dpsi_dr = np.gradient(flux, z, r, edge_order=2)
    safe_r = np.where(r_grid == 0.0, np.inf, r_grid)
    b_r = -dpsi_dz / safe_r
    b_z = dpsi_dr / safe_r
    b_r[:, 0] = 0.0
    b_z[:, 0] = -axial_field_t * np.cos(np.pi * z / length_m) ** 2

    return AxisymmetricFieldMap(
        r_grid=r_grid,
        z_grid=z_grid,
        flux_webers=flux,
        b_r_t=b_r,
        b_z_t=b_z,
    )
