"""
RPST Topology - Eq 0c
Topological charge and winding number computation on Z_p.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class TopologicalCharge:
    """Conserved topological quantum number."""

    winding: int  # Integer winding number W
    loop_path: List[int]  # Indices defining the loop
    total_phase: float  # Accumulated phase around loop (radians)


def minimal_signed_difference(a: int, b: int, p: int) -> int:
    """
    Minimal signed difference on Z_p, mapped to roughly [-(p-1)/2, (p-1)/2].
    """

    diff = (int(a) - int(b)) % int(p)
    half_p = int(p) // 2
    if diff > half_p:
        return diff - int(p)
    return diff


def compute_winding_number(q_values: np.ndarray, loop_indices: List[int], p: int) -> TopologicalCharge:
    """
    Compute topological winding number for a closed loop C (Eq 0c).

        W = (1/p) Σ_{(i,j)∈C} Δ(q_j - q_i)

    Args:
        q_values: 1D array of q values in Z_p
        loop_indices: node indices defining a closed loop
        p: prime modulus
    """

    q_values = np.asarray(q_values, dtype=int) % int(p)
    total_delta = 0
    n = len(loop_indices)
    if n < 2:
        return TopologicalCharge(winding=0, loop_path=list(loop_indices), total_phase=0.0)

    for k in range(n):
        i = loop_indices[k]
        j = loop_indices[(k + 1) % n]
        delta = minimal_signed_difference(int(q_values[j]), int(q_values[i]), int(p))
        total_delta += int(delta)

    W_continuous = total_delta / int(p)
    W = int(round(W_continuous))

    # Basic quantization sanity
    if abs(W_continuous - W) > 0.01:
        raise ValueError(f"Winding number not quantized: {W_continuous}")

    return TopologicalCharge(winding=W, loop_path=list(loop_indices), total_phase=float(2 * np.pi * W_continuous))


def find_soliton_cores(q_field: np.ndarray, p: int, threshold: float = 0.3) -> List[Tuple[int, int, int]]:
    """
    Find soliton cores (topological defects) in a 2D q field by checking plaquette windings.

    Returns:
        list of (i, j, winding) for plaquettes with |W| > threshold
    """

    q_field = np.asarray(q_field, dtype=int) % int(p)
    Nx, Ny = q_field.shape
    solitons: List[Tuple[int, int, int]] = []

    for i in range(Nx - 1):
        for j in range(Ny - 1):
            q_corners = [
                q_field[i, j],
                q_field[i + 1, j],
                q_field[i + 1, j + 1],
                q_field[i, j + 1],
            ]
            total_delta = 0
            for k in range(4):
                total_delta += minimal_signed_difference(
                    int(q_corners[(k + 1) % 4]), int(q_corners[k]), int(p)
                )
            W = total_delta / int(p)
            if abs(W) > float(threshold):
                solitons.append((i, j, int(round(W))))

    return solitons


def verify_charge_conservation(
    initial_state: np.ndarray, final_state: np.ndarray, p: int, loops: List[List[int]]
) -> Tuple[bool, float]:
    """Verify winding is conserved for the provided loops."""

    max_dev = 0.0
    for loop in loops:
        W0 = compute_winding_number(initial_state, loop, int(p)).winding
        W1 = compute_winding_number(final_state, loop, int(p)).winding
        max_dev = max(max_dev, float(abs(W0 - W1)))
    return max_dev < 0.5, max_dev


