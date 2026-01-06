"""
RPST Coarse Graining - Eq 0d/0e scaffolding

Provides a lightweight mapping from discrete Z_p states to a smooth continuum
field and a simple wave-equation residual check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .substrate import SubstrateState


@dataclass
class CoarseGraining:
    """
    Coarse-grain discrete q samples at spatial positions into a smooth field.

    positions: shape (N, d)
    coarse_scale: Gaussian kernel width in the same units as positions
    """

    positions: np.ndarray
    coarse_scale: float = 0.5

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2:
            raise ValueError("positions must be (N, d)")
        if self.coarse_scale <= 0:
            raise ValueError("coarse_scale must be > 0")

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        r2 = float(np.sum((x - y) ** 2))
        return float(np.exp(-r2 / (2 * self.coarse_scale**2)))

    def field_from_state(self, state: SubstrateState, eval_points: np.ndarray) -> np.ndarray:
        """
        Produce a smooth real field by kernel-averaging q mapped to [-0.5, 0.5).
        """

        q = np.asarray(state.q, dtype=int) % int(state.p)
        q_real = (q / int(state.p)) - 0.5
        eval_points = np.asarray(eval_points, dtype=float)
        if eval_points.ndim == 1:
            eval_points = eval_points.reshape(-1, 1)

        out = np.zeros(eval_points.shape[0], dtype=float)
        for i, xp in enumerate(eval_points):
            w = np.array([self._kernel(xp, xi) for xi in self.positions], dtype=float)
            sw = float(np.sum(w))
            if sw == 0:
                out[i] = 0.0
            else:
                out[i] = float(np.sum(w * q_real) / sw)
        return out

    def verify_wave_equation(
        self,
        trajectory: List[SubstrateState],
        dt: float,
        eval_points: np.ndarray,
        c: float = 1.0,
        tolerance: float = 1.0,
    ) -> Tuple[bool, float]:
        """
        Check a crude discrete wave equation residual:

          u_tt - c^2 Δ u ≈ 0

        using finite differences in time and a 1D second-difference Laplacian
        if eval_points are 1D and ordered.
        """

        if len(trajectory) < 3:
            return False, float("inf")

        u0 = self.field_from_state(trajectory[0], eval_points)
        u1 = self.field_from_state(trajectory[1], eval_points)
        u2 = self.field_from_state(trajectory[2], eval_points)

        u_tt = (u2 - 2 * u1 + u0) / (float(dt) ** 2)

        # Simple Laplacian: assume eval_points are 1D uniform-ish
        x = np.asarray(eval_points, dtype=float).reshape(-1)
        if x.size < 3:
            return False, float("inf")
        dx = float(np.mean(np.diff(x)))
        if dx == 0:
            return False, float("inf")

        u_xx = np.zeros_like(u1)
        u_xx[1:-1] = (u1[2:] - 2 * u1[1:-1] + u1[:-2]) / (dx**2)
        u_xx[0] = u_xx[1]
        u_xx[-1] = u_xx[-2]

        residual = u_tt - (float(c) ** 2) * u_xx
        max_res = float(np.max(np.abs(residual)))
        return max_res < float(tolerance), max_res


def verify_wave_equation(
    trajectory: List[SubstrateState],
    positions: np.ndarray,
    eval_points: np.ndarray,
    dt: float,
    coarse_scale: float = 0.5,
    c: float = 1.0,
    tolerance: float = 1.0,
) -> Tuple[bool, float]:
    """Convenience wrapper."""

    cg = CoarseGraining(positions=np.asarray(positions, dtype=float), coarse_scale=float(coarse_scale))
    return cg.verify_wave_equation(trajectory=trajectory, dt=dt, eval_points=eval_points, c=c, tolerance=tolerance)


