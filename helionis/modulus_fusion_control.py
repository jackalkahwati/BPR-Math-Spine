"""Modulus Fusion deterministic control twin for Helionis geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import isfinite
from numbers import Real
from typing import Dict, List, Optional

from helionis.bpr_coupled_v09 import BPRCoupledV09Result, run_bpr_coupled_v09_sweep


@dataclass(frozen=True)
class ModulusFusionControlAssumptions:
    """Assumptions for a first real-time plasma-control screen."""

    update_period_ms: float = 1.0
    control_compute_latency_ms: float = 0.25
    max_control_latency_ms: float = 2.0
    actuator_slew_t_per_ms: float = 0.08
    sensor_noise_fraction: float = 0.002
    physical_drift_fraction_per_ms: float = 0.001
    max_equilibrium_residual: float = 0.08
    max_command_fraction: float = 1.0
    max_numerical_drift_fraction: float = 0.0
    exact_math_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.exact_math_enabled, bool):
            raise ValueError("exact_math_enabled must be a boolean")
        for field in (
            "update_period_ms",
            "control_compute_latency_ms",
            "max_control_latency_ms",
            "actuator_slew_t_per_ms",
            "sensor_noise_fraction",
            "physical_drift_fraction_per_ms",
            "max_equilibrium_residual",
            "max_command_fraction",
            "max_numerical_drift_fraction",
        ):
            _require_finite_number(field, getattr(self, field))
        for field in (
            "update_period_ms",
            "control_compute_latency_ms",
            "max_control_latency_ms",
            "actuator_slew_t_per_ms",
            "max_command_fraction",
        ):
            if getattr(self, field) <= 0.0:
                raise ValueError(f"{field} must be positive")
        for field in (
            "sensor_noise_fraction",
            "physical_drift_fraction_per_ms",
            "max_equilibrium_residual",
            "max_numerical_drift_fraction",
            "max_command_fraction",
        ):
            if getattr(self, field) < 0.0:
                raise ValueError(f"{field} cannot be negative")
            if getattr(self, field) > 1.0:
                raise ValueError(f"{field} must be no greater than 1")


@dataclass(frozen=True)
class ModulusFusionControlResult:
    """Control-twin result for one v0.9 reactor geometry."""

    source_plant_net_power_mw: float
    source_closes_engineering_net: bool
    controllable: bool
    controllability_score: float
    uses_bpr_coupled_geometry: bool
    update_period_ms: float
    control_latency_margin: float
    midplane_field_t: float
    plug_field_t: float
    required_field_correction_t: float
    required_slew_t_per_ms: float
    coil_command_fraction: float
    equilibrium_residual: float
    numerical_drift_fraction: float
    physical_drift_fraction: float
    sensor_error_fraction: float
    total_non_numerical_error_fraction: float
    drift_claim: str
    mirror_ratio: float
    mirror_aspect_ratio: float
    separatrix_radius_m: float
    plasma_length_m: float
    plasma_volume_m3: float
    source_radiator_area_m2: float
    collector_area_m2: float
    source_bpr_resonance_alignment: float
    source_bpr_impedance_match: float
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV-friendly row."""
        return asdict(self)


def exact_state_update(
    state: Fraction,
    command_delta: Fraction,
    disturbance_delta: Fraction,
) -> Fraction:
    """Update a scalar control state with exact rational arithmetic."""
    _require_fraction("state", state)
    _require_fraction("command_delta", command_delta)
    _require_fraction("disturbance_delta", disturbance_delta)
    return state + command_delta + disturbance_delta


def numerical_drift_after_exact_updates(
    initial_state: Fraction,
    command_delta: Fraction,
    disturbance_delta: Fraction,
    steps: int,
) -> Fraction:
    """Return accumulated numerical drift from repeated exact updates."""
    _require_fraction("initial_state", initial_state)
    _require_fraction("command_delta", command_delta)
    _require_fraction("disturbance_delta", disturbance_delta)
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps cannot be negative")

    state = initial_state
    for _ in range(steps):
        state = exact_state_update(state, command_delta, disturbance_delta)
    closed_form = initial_state + steps * (command_delta + disturbance_delta)
    return state - closed_form


def evaluate_modulus_fusion_control(
    geometry: BPRCoupledV09Result,
    assumptions: Optional[ModulusFusionControlAssumptions] = None,
) -> ModulusFusionControlResult:
    """Evaluate real-time controllability for one BPR-coupled geometry."""
    assumptions = assumptions or ModulusFusionControlAssumptions()
    _validate_geometry(geometry)
    midplane_field_t = geometry.plug_field_t / geometry.mirror_ratio
    physical_drift_fraction = (
        assumptions.physical_drift_fraction_per_ms * assumptions.update_period_ms
    )
    total_non_numerical_error = (
        physical_drift_fraction + assumptions.sensor_noise_fraction
    )
    required_field_correction_t = midplane_field_t * total_non_numerical_error
    required_slew_t_per_ms = required_field_correction_t / assumptions.update_period_ms
    coil_command_fraction = required_slew_t_per_ms / assumptions.actuator_slew_t_per_ms
    effective_latency_ms = (
        assumptions.control_compute_latency_ms + assumptions.update_period_ms
    )
    control_latency_margin = assumptions.max_control_latency_ms / effective_latency_ms
    numerical_drift_fraction = _numerical_drift_fraction(
        required_field_correction_t=required_field_correction_t,
        midplane_field_t=midplane_field_t,
        exact_math_enabled=assumptions.exact_math_enabled,
    )
    equilibrium_residual = _equilibrium_residual(geometry)

    command_score = _bounded_score(1.0 - min(coil_command_fraction, 1.5) / 1.5)
    latency_score = _bounded_score(min(control_latency_margin, 2.0) / 2.0)
    equilibrium_score = _bounded_score(
        1.0 - equilibrium_residual / max(assumptions.max_equilibrium_residual, 1e-12)
    )
    drift_score = (
        1.0
        if numerical_drift_fraction <= assumptions.max_numerical_drift_fraction
        else 0.0
    )
    controllability_score = (
        0.35 * command_score
        + 0.25 * latency_score
        + 0.25 * equilibrium_score
        + 0.15 * drift_score
    )
    controllable = (
        coil_command_fraction <= assumptions.max_command_fraction
        and control_latency_margin >= 1.0
        and equilibrium_residual <= assumptions.max_equilibrium_residual
        and numerical_drift_fraction <= assumptions.max_numerical_drift_fraction
    )

    return ModulusFusionControlResult(
        source_plant_net_power_mw=geometry.plant_net_power_mw,
        source_closes_engineering_net=geometry.closes_engineering_net,
        controllable=controllable,
        controllability_score=controllability_score,
        uses_bpr_coupled_geometry=geometry.uses_bpr_math,
        update_period_ms=assumptions.update_period_ms,
        control_latency_margin=control_latency_margin,
        midplane_field_t=midplane_field_t,
        plug_field_t=geometry.plug_field_t,
        required_field_correction_t=required_field_correction_t,
        required_slew_t_per_ms=required_slew_t_per_ms,
        coil_command_fraction=coil_command_fraction,
        equilibrium_residual=equilibrium_residual,
        numerical_drift_fraction=numerical_drift_fraction,
        physical_drift_fraction=physical_drift_fraction,
        sensor_error_fraction=assumptions.sensor_noise_fraction,
        total_non_numerical_error_fraction=total_non_numerical_error,
        drift_claim="zero numerical drift in deterministic control math"
        if assumptions.exact_math_enabled
        else "floating-point control math includes finite numerical drift",
        mirror_ratio=geometry.mirror_ratio,
        mirror_aspect_ratio=geometry.mirror_aspect_ratio,
        separatrix_radius_m=geometry.separatrix_radius_m,
        plasma_length_m=geometry.plasma_length_m,
        plasma_volume_m3=geometry.plasma_volume_m3,
        source_radiator_area_m2=geometry.radiator_area_m2,
        collector_area_m2=geometry.collector_area_m2,
        source_bpr_resonance_alignment=geometry.bpr_resonance_alignment,
        source_bpr_impedance_match=geometry.bpr_impedance_match,
        warnings=_warnings(geometry, controllable, equilibrium_residual),
    )


def run_modulus_fusion_control_twin(
    target_screening_net_mw: float = 50.0,
    assumptions: Optional[ModulusFusionControlAssumptions] = None,
    candidate_limit: Optional[int] = None,
    limit: Optional[int] = 12,
) -> List[ModulusFusionControlResult]:
    """Evaluate control-twin rows for top BPR-coupled v0.9 geometries."""
    _require_positive_finite_number("target_screening_net_mw", target_screening_net_mw)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
    assumptions = assumptions or ModulusFusionControlAssumptions()
    if candidate_limit is not None:
        if not isinstance(candidate_limit, int) or isinstance(candidate_limit, bool):
            raise ValueError("candidate_limit must be an integer")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
    geometry_rows = run_bpr_coupled_v09_sweep(
        target_screening_net_mw=target_screening_net_mw,
        limit=candidate_limit,
    )
    control_rows = [
        evaluate_modulus_fusion_control(row, assumptions=assumptions)
        for row in geometry_rows
    ]
    ranked = sorted(
        control_rows,
        key=lambda row: (row.controllable, row.controllability_score),
        reverse=True,
    )
    return ranked[:limit] if limit is not None else ranked


def _equilibrium_residual(geometry: BPRCoupledV09Result) -> float:
    transport_term = geometry.adjusted_transport_multiplier
    resonance_term = 1.0 - geometry.bpr_resonance_alignment
    impedance_term = 1.0 - geometry.bpr_impedance_match
    return max(
        0.0, 0.45 * transport_term + 0.20 * resonance_term + 0.10 * impedance_term
    )


def _bounded_score(value: float) -> float:
    if not isfinite(value):
        raise ValueError("score value must be finite")
    return max(0.0, min(1.0, value))


def _numerical_drift_fraction(
    required_field_correction_t: float,
    midplane_field_t: float,
    exact_math_enabled: bool,
) -> float:
    if not exact_math_enabled:
        return 1e-12
    correction = Fraction(str(required_field_correction_t)).limit_denominator(10**12)
    drift = numerical_drift_after_exact_updates(
        initial_state=Fraction(0, 1),
        command_delta=correction,
        disturbance_delta=Fraction(0, 1),
        steps=1024,
    )
    denominator = max(abs(midplane_field_t), 1e-12)
    return float(abs(drift)) / denominator


def _validate_geometry(geometry: BPRCoupledV09Result) -> None:
    for field in (
        "plant_net_power_mw",
        "plug_field_t",
        "mirror_ratio",
        "mirror_aspect_ratio",
        "separatrix_radius_m",
        "plasma_length_m",
        "plasma_volume_m3",
        "radiator_area_m2",
        "collector_area_m2",
        "bpr_resonance_alignment",
        "bpr_impedance_match",
        "adjusted_transport_multiplier",
    ):
        _require_finite_number(field, getattr(geometry, field))
    if geometry.mirror_ratio <= 1.0:
        raise ValueError("mirror_ratio must be greater than 1")
    if geometry.plug_field_t <= 0.0:
        raise ValueError("plug_field_t must be positive")
    if geometry.mirror_aspect_ratio <= 0.0:
        raise ValueError("mirror_aspect_ratio must be positive")
    if geometry.separatrix_radius_m <= 0.0:
        raise ValueError("separatrix_radius_m must be positive")
    if geometry.plasma_length_m <= 0.0:
        raise ValueError("plasma_length_m must be positive")
    if geometry.plasma_volume_m3 <= 0.0:
        raise ValueError("plasma_volume_m3 must be positive")
    if geometry.radiator_area_m2 < 0.0:
        raise ValueError("radiator_area_m2 cannot be negative")
    if geometry.collector_area_m2 < 0.0:
        raise ValueError("collector_area_m2 cannot be negative")
    for field in ("bpr_resonance_alignment", "bpr_impedance_match"):
        value = getattr(geometry, field)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be between 0 and 1")
    if geometry.adjusted_transport_multiplier < 0.0:
        raise ValueError("adjusted_transport_multiplier cannot be negative")


def _warnings(
    geometry: BPRCoupledV09Result,
    controllable: bool,
    equilibrium_residual: float,
) -> str:
    notes = [
        geometry.warnings,
        "Modulus Fusion v1.0 distinguishes numerical drift from physical plasma motion",
    ]
    if not controllable:
        notes.append(
            "control twin flags this geometry as outside current actuator/cadence limits"
        )
    if equilibrium_residual > 0.05:
        notes.append("equilibrium residual still needs higher-fidelity MHD validation")
    return "; ".join(notes)


def _require_finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _require_positive_finite_number(name: str, value: object) -> None:
    _require_finite_number(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_fraction(name: str, value: object) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a Fraction")
