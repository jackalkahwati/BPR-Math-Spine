"""Bridge from Helionis screening models into BPR Math Spine primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from typing import Dict

import numpy as np

from bpr.impedance import TopologicalImpedance
from bpr.resonance import load_riemann_zeros


@dataclass(frozen=True)
class BPRBridgeInputs:
    """Inputs for BPR-derived correction factors in Helionis screens."""

    mirror_ratio: float
    radius_m: float
    length_m: float
    base_transport_multiplier: float
    base_direct_conversion_efficiency: float

    def __post_init__(self) -> None:
        for field in (
            "mirror_ratio",
            "radius_m",
            "length_m",
            "base_transport_multiplier",
            "base_direct_conversion_efficiency",
        ):
            _require_finite_number(field, getattr(self, field))
        if self.mirror_ratio <= 1.0:
            raise ValueError("mirror_ratio must be greater than 1")
        if self.radius_m <= 0 or self.length_m <= 0:
            raise ValueError("radius_m and length_m must be positive")
        if self.base_transport_multiplier < 0:
            raise ValueError("base_transport_multiplier cannot be negative")
        if not 0.0 <= self.base_direct_conversion_efficiency <= 1.0:
            raise ValueError(
                "base_direct_conversion_efficiency must be between 0 and 1"
            )


@dataclass(frozen=True)
class BPRBridgeFactors:
    """Bounded BPR-derived factors used by Helionis."""

    impedance_match: float
    resonance_alignment: float
    transport_multiplier: float
    direct_conversion_multiplier: float
    source_modules: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV-friendly row."""
        return asdict(self)


def topological_impedance_match(
    effective_winding: float,
    critical_winding: float = 10.0,
) -> float:
    """Return BPR topological EM-coupling proxy from impedance mismatch."""
    _require_finite_number("effective_winding", effective_winding)
    _require_finite_number("critical_winding", critical_winding)
    if effective_winding < 0:
        raise ValueError("effective_winding cannot be negative")
    if critical_winding <= 0:
        raise ValueError("critical_winding must be positive")
    impedance = TopologicalImpedance(W_c=critical_winding)
    return float(impedance.em_coupling(effective_winding))


def resonance_alignment_factor(
    radius_m: float,
    length_m: float,
    n_zeros: int = 20,
) -> float:
    """Return geometry alignment with BPR/Riemann-zero resonance spacings."""
    _require_finite_number("radius_m", radius_m)
    _require_finite_number("length_m", length_m)
    if not isinstance(n_zeros, int) or isinstance(n_zeros, bool):
        raise ValueError("n_zeros must be an integer")
    if n_zeros <= 0:
        raise ValueError("n_zeros must be positive")
    if radius_m <= 0 or length_m <= 0:
        raise ValueError("radius_m and length_m must be positive")
    zeros = load_riemann_zeros(n_zeros)
    normalized = zeros / zeros[0]
    geometry_ratio = length_m / (2.0 * radius_m)
    if not isfinite(geometry_ratio) or geometry_ratio <= 0.0:
        raise ValueError("length_m / radius_m produced a non-finite ratio")
    relative_errors = np.abs(normalized - geometry_ratio) / geometry_ratio
    nearest_error = float(np.min(relative_errors))
    factor = 1.0 / (1.0 + 4.0 * nearest_error)
    if not isfinite(factor):
        raise ValueError("resonance alignment produced a non-finite factor")
    return factor


def bpr_bridge_factors(inputs: BPRBridgeInputs) -> BPRBridgeFactors:
    """Compute bounded BPR factors for Helionis mirror/nozzle screening."""
    impedance_match = topological_impedance_match(inputs.mirror_ratio)
    resonance_alignment = resonance_alignment_factor(inputs.radius_m, inputs.length_m)

    # Conservative bounded usage: BPR factors can trim transport or improve
    # coupling modestly, but cannot create unbounded gains in the screening model.
    mismatch = 1.0 - impedance_match
    transport_multiplier = max(
        0.70,
        (1.0 - 0.20 * resonance_alignment) * (1.0 - 0.15 * mismatch),
    )
    direct_conversion_multiplier = min(
        1.05,
        1.0 / inputs.base_direct_conversion_efficiency
        if inputs.base_direct_conversion_efficiency > 0.0
        else 1.05,
        1.0 + 0.03 * resonance_alignment + 0.02 * impedance_match,
    )

    return BPRBridgeFactors(
        impedance_match=impedance_match,
        resonance_alignment=resonance_alignment,
        transport_multiplier=transport_multiplier,
        direct_conversion_multiplier=direct_conversion_multiplier,
        source_modules=(
            "bpr.impedance.TopologicalImpedance; bpr.resonance.load_riemann_zeros"
        ),
    )


def _require_finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
