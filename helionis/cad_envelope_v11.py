"""LunarFire v1.1 control-constrained parametric CAD envelope."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, pi, sqrt
from numbers import Real
from typing import Dict, List, Optional

from helionis.modulus_fusion_control import (
    ModulusFusionControlResult,
    run_modulus_fusion_control_twin,
)


@dataclass(frozen=True)
class CADEnvelopeV11Assumptions:
    """Packaging assumptions for the first CAD envelope."""

    chamber_clearance_m: float = 0.60
    service_clearance_m: float = 0.80
    min_coil_radial_build_m: float = 0.35
    coil_radial_build_m_per_t: float = 0.08
    plug_coil_build_multiplier: float = 1.20
    nozzle_length_factor: float = 1.25
    min_nozzle_length_m: float = 3.0
    collector_area_margin: float = 0.15
    collector_length_fraction_of_nozzle: float = 0.85
    radiator_area_margin: float = 0.10
    max_machine_length_m: float = 80.0
    max_outer_radius_m: float = 18.0
    max_radiator_area_m2: float = 45_000.0
    max_radiator_wing_span_each_m: float = 500.0

    def __post_init__(self) -> None:
        for field in (
            "chamber_clearance_m",
            "service_clearance_m",
            "min_coil_radial_build_m",
            "coil_radial_build_m_per_t",
            "plug_coil_build_multiplier",
            "nozzle_length_factor",
            "min_nozzle_length_m",
            "collector_area_margin",
            "collector_length_fraction_of_nozzle",
            "radiator_area_margin",
            "max_machine_length_m",
            "max_outer_radius_m",
            "max_radiator_area_m2",
            "max_radiator_wing_span_each_m",
        ):
            _require_finite_number(field, getattr(self, field))
        for field in (
            "chamber_clearance_m",
            "service_clearance_m",
            "min_coil_radial_build_m",
            "coil_radial_build_m_per_t",
            "plug_coil_build_multiplier",
            "nozzle_length_factor",
            "min_nozzle_length_m",
            "collector_length_fraction_of_nozzle",
            "max_machine_length_m",
            "max_outer_radius_m",
            "max_radiator_area_m2",
            "max_radiator_wing_span_each_m",
        ):
            if getattr(self, field) <= 0.0:
                raise ValueError(f"{field} must be positive")
        for field in ("collector_area_margin", "radiator_area_margin"):
            if getattr(self, field) < 0.0:
                raise ValueError(f"{field} cannot be negative")
        if self.collector_length_fraction_of_nozzle > 1.0:
            raise ValueError(
                "collector_length_fraction_of_nozzle must be no greater than 1"
            )


@dataclass(frozen=True)
class CADEnvelopeV11Result:
    """One parametric CAD envelope row."""

    source_controllability_score: float
    source_plant_net_power_mw: float
    source_closes_engineering_net: bool
    cad_ready: bool
    cad_readiness_score: float
    uses_modulus_control: bool
    plasma_radius_m: float
    plasma_length_m: float
    plasma_volume_m3: float
    chamber_inner_radius_m: float
    coil_radial_build_m: float
    midplane_coil_radius_m: float
    plug_coil_radius_m: float
    nozzle_length_m: float
    collector_length_m: float
    collector_radius_m: float
    collector_surface_area_m2: float
    radiator_area_m2: float
    radiator_wing_span_each_m: float
    outer_radius_m: float
    machine_length_m: float
    mirror_ratio: float
    mirror_aspect_ratio: float
    coil_command_fraction: float
    equilibrium_residual: float
    warnings: str
    blockers: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV-friendly row."""
        return asdict(self)


def evaluate_cad_envelope(
    control: ModulusFusionControlResult,
    assumptions: Optional[CADEnvelopeV11Assumptions] = None,
) -> CADEnvelopeV11Result:
    """Build a parametric CAD envelope around one control-twin row."""
    assumptions = assumptions or CADEnvelopeV11Assumptions()
    _validate_control_row(control)

    plasma_radius = control.separatrix_radius_m
    plasma_length = control.plasma_length_m
    chamber_inner_radius = plasma_radius + assumptions.chamber_clearance_m
    coil_radial_build = max(
        assumptions.min_coil_radial_build_m,
        assumptions.coil_radial_build_m_per_t * control.midplane_field_t,
    )
    midplane_coil_radius = chamber_inner_radius + coil_radial_build
    plug_coil_radius = chamber_inner_radius + (
        coil_radial_build * assumptions.plug_coil_build_multiplier
    )
    nozzle_length = max(
        assumptions.min_nozzle_length_m,
        assumptions.nozzle_length_factor * plasma_radius * control.mirror_ratio,
    )
    collector_length = nozzle_length * assumptions.collector_length_fraction_of_nozzle
    collector_surface_area = control.collector_area_m2 * (
        1.0 + assumptions.collector_area_margin
    )
    collector_radius = max(
        chamber_inner_radius,
        collector_surface_area / max(2.0 * pi * collector_length, 1e-12),
    )
    radiator_area = control.source_radiator_area_m2 * (
        1.0 + assumptions.radiator_area_margin
    )
    machine_length = (
        plasma_length + 2.0 * nozzle_length + 2.0 * assumptions.service_clearance_m
    )
    outer_radius = (
        max(plug_coil_radius, collector_radius) + assumptions.service_clearance_m
    )
    radiator_wing_span_each = radiator_area / max(2.0 * machine_length, 1e-12)

    length_score = _bounded_score(
        1.0 - machine_length / assumptions.max_machine_length_m
    )
    radius_score = _bounded_score(1.0 - outer_radius / assumptions.max_outer_radius_m)
    radiator_area_score = _bounded_score(
        1.0 - radiator_area / assumptions.max_radiator_area_m2
    )
    wing_span_score = _bounded_score(
        1.0 - radiator_wing_span_each / assumptions.max_radiator_wing_span_each_m
    )
    radiator_score = min(radiator_area_score, wing_span_score)
    cad_readiness_score = (
        0.35 * control.controllability_score
        + 0.25 * length_score
        + 0.20 * radius_score
        + 0.20 * radiator_score
    )
    failures = _constraint_failures(
        machine_length=machine_length,
        outer_radius=outer_radius,
        radiator_area=radiator_area,
        radiator_wing_span_each=radiator_wing_span_each,
        assumptions=assumptions,
    )
    plant_closes_engineering_net = control.source_plant_net_power_mw > 0.0
    cad_ready = control.controllable and plant_closes_engineering_net and not failures

    return CADEnvelopeV11Result(
        source_controllability_score=control.controllability_score,
        source_plant_net_power_mw=control.source_plant_net_power_mw,
        source_closes_engineering_net=control.source_closes_engineering_net,
        cad_ready=cad_ready,
        cad_readiness_score=_bounded_score(cad_readiness_score),
        uses_modulus_control=True,
        plasma_radius_m=plasma_radius,
        plasma_length_m=plasma_length,
        plasma_volume_m3=control.plasma_volume_m3,
        chamber_inner_radius_m=chamber_inner_radius,
        coil_radial_build_m=coil_radial_build,
        midplane_coil_radius_m=midplane_coil_radius,
        plug_coil_radius_m=plug_coil_radius,
        nozzle_length_m=nozzle_length,
        collector_length_m=collector_length,
        collector_radius_m=collector_radius,
        collector_surface_area_m2=collector_surface_area,
        radiator_area_m2=radiator_area,
        radiator_wing_span_each_m=radiator_wing_span_each,
        outer_radius_m=outer_radius,
        machine_length_m=machine_length,
        mirror_ratio=control.mirror_ratio,
        mirror_aspect_ratio=control.mirror_aspect_ratio,
        coil_command_fraction=control.coil_command_fraction,
        equilibrium_residual=control.equilibrium_residual,
        warnings=_warnings(control, failures),
        blockers=_blockers(control, failures),
    )


def run_cad_envelope_v11(
    target_screening_net_mw: float = 50.0,
    assumptions: Optional[CADEnvelopeV11Assumptions] = None,
    limit: Optional[int] = 12,
) -> List[CADEnvelopeV11Result]:
    """Evaluate CAD envelopes for Modulus Fusion control-twin rows."""
    _require_positive_finite_number("target_screening_net_mw", target_screening_net_mw)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
    assumptions = assumptions or CADEnvelopeV11Assumptions()
    controls = run_modulus_fusion_control_twin(
        target_screening_net_mw=target_screening_net_mw,
        limit=None,
    )
    rows = [
        evaluate_cad_envelope(control, assumptions=assumptions) for control in controls
    ]
    ranked = sorted(
        rows,
        key=lambda row: (row.cad_ready, row.cad_readiness_score),
        reverse=True,
    )
    return ranked[:limit] if limit is not None else ranked


def _constraint_failures(
    machine_length: float,
    outer_radius: float,
    radiator_area: float,
    radiator_wing_span_each: float,
    assumptions: CADEnvelopeV11Assumptions,
) -> List[str]:
    failures: List[str] = []
    if machine_length > assumptions.max_machine_length_m:
        failures.append("machine length exceeds CAD envelope limit")
    if outer_radius > assumptions.max_outer_radius_m:
        failures.append("outer radius exceeds CAD envelope limit")
    if radiator_area > assumptions.max_radiator_area_m2:
        failures.append("radiator area exceeds CAD envelope limit")
    if radiator_wing_span_each > assumptions.max_radiator_wing_span_each_m:
        failures.append("radiator wing span exceeds CAD envelope limit")
    return failures


def _warnings(control: ModulusFusionControlResult, failures: List[str]) -> str:
    notes = [
        control.warnings,
        "v1.1 parametric CAD envelope is control-constrained, not detailed CAD",
    ]
    notes.extend(failures)
    if control.source_plant_net_power_mw <= 0.0:
        notes.append("source plant does not close engineering net")
    return "; ".join(notes)


def _blockers(control: ModulusFusionControlResult, failures: List[str]) -> str:
    blockers = list(failures)
    if not control.controllable:
        blockers.append("control row is not controllable")
    if control.source_plant_net_power_mw <= 0.0:
        blockers.append("source plant is net-negative")
    return "; ".join(blockers) if blockers else "none"


def _validate_control_row(control: ModulusFusionControlResult) -> None:
    if not isinstance(control.controllable, bool):
        raise ValueError("controllable must be a boolean")
    if not isinstance(control.source_closes_engineering_net, bool):
        raise ValueError("source_closes_engineering_net must be a boolean")
    if control.source_closes_engineering_net != (
        control.source_plant_net_power_mw > 0.0
    ):
        raise ValueError(
            "source_closes_engineering_net is inconsistent with plant-net power"
        )
    for field in (
        "controllability_score",
        "source_plant_net_power_mw",
        "separatrix_radius_m",
        "plasma_length_m",
        "plasma_volume_m3",
        "midplane_field_t",
        "source_radiator_area_m2",
        "collector_area_m2",
        "mirror_ratio",
        "mirror_aspect_ratio",
        "coil_command_fraction",
        "equilibrium_residual",
    ):
        _require_finite_number(field, getattr(control, field))
    if not 0.0 <= control.controllability_score <= 1.0:
        raise ValueError("controllability_score must be between 0 and 1")
    if control.separatrix_radius_m <= 0.0:
        raise ValueError("separatrix_radius_m must be positive")
    if control.plasma_length_m <= 0.0:
        raise ValueError("plasma_length_m must be positive")
    if control.plasma_volume_m3 <= 0.0:
        raise ValueError("plasma_volume_m3 must be positive")
    if control.midplane_field_t <= 0.0:
        raise ValueError("midplane_field_t must be positive")
    if control.source_radiator_area_m2 < 0.0:
        raise ValueError("source_radiator_area_m2 cannot be negative")
    if control.collector_area_m2 < 0.0:
        raise ValueError("collector_area_m2 cannot be negative")
    if control.mirror_ratio <= 1.0:
        raise ValueError("mirror_ratio must be greater than 1")
    if control.mirror_aspect_ratio <= 0.0:
        raise ValueError("mirror_aspect_ratio must be positive")
    if control.coil_command_fraction < 0.0:
        raise ValueError("coil_command_fraction cannot be negative")
    if control.equilibrium_residual < 0.0:
        raise ValueError("equilibrium_residual cannot be negative")


def _bounded_score(value: float) -> float:
    if not isfinite(value):
        raise ValueError("score value must be finite")
    return max(0.0, min(1.0, value))


def _require_finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _require_positive_finite_number(name: str, value: object) -> None:
    _require_finite_number(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
