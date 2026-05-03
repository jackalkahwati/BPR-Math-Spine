"""LunarFire v1.2 thermal packaging recovery analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, log2
from numbers import Real
from typing import Dict, List, Optional, Sequence

from helionis.cad_envelope_v11 import CADEnvelopeV11Result, run_cad_envelope_v11
from helionis.engineering_net import STEFAN_BOLTZMANN_W_M2_K4


DEFAULT_DIRECT_HEAT_RECOVERY_FRACTIONS = (0.0, 0.01, 0.02, 0.04)
DEFAULT_RADIATOR_TEMPERATURES_K = (800.0, 1000.0, 1200.0, 1400.0)
DEFAULT_TOPOLOGY_PACKING_FACTORS = (1.0, 2.0, 4.0)


@dataclass(frozen=True)
class ThermalRecoveryV12Assumptions:
    """One thermal-packaging recovery recipe."""

    direct_heat_recovery_fraction: float = 0.0
    radiator_temperature_k: float = 800.0
    radiator_emissivity: float = 0.85
    topology_packing_factor: float = 1.0
    max_wing_span_each_m: float = 500.0
    max_radiator_area_m2: float = 45_000.0
    max_machine_length_m: float = 80.0
    max_outer_radius_m: float = 18.0
    base_radiator_temperature_k: float = 800.0
    base_radiator_emissivity: float = 0.85
    source_radiator_area_margin: float = 0.10
    max_direct_heat_recovery_fraction: float = 0.05

    def __post_init__(self) -> None:
        for field in (
            "direct_heat_recovery_fraction",
            "radiator_temperature_k",
            "radiator_emissivity",
            "topology_packing_factor",
            "max_wing_span_each_m",
            "max_radiator_area_m2",
            "max_machine_length_m",
            "max_outer_radius_m",
            "base_radiator_temperature_k",
            "base_radiator_emissivity",
            "source_radiator_area_margin",
            "max_direct_heat_recovery_fraction",
        ):
            _require_finite_number(field, getattr(self, field))
        if (
            not 0.0
            <= self.direct_heat_recovery_fraction
            <= self.max_direct_heat_recovery_fraction
        ):
            raise ValueError(
                "direct_heat_recovery_fraction must be between 0 and "
                "max_direct_heat_recovery_fraction"
            )
        for field in (
            "radiator_temperature_k",
            "topology_packing_factor",
            "max_wing_span_each_m",
            "max_radiator_area_m2",
            "max_machine_length_m",
            "max_outer_radius_m",
            "base_radiator_temperature_k",
            "max_direct_heat_recovery_fraction",
        ):
            if getattr(self, field) <= 0.0:
                raise ValueError(f"{field} must be positive")
        if self.source_radiator_area_margin < 0.0:
            raise ValueError("source_radiator_area_margin cannot be negative")
        for field in ("radiator_emissivity", "base_radiator_emissivity"):
            if not 0.0 < getattr(self, field) <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")


@dataclass(frozen=True)
class ThermalRecoveryV12Result:
    """One thermal recovery row."""

    source_cad_readiness_score: float
    source_plant_net_power_mw: float
    plant_net_power_mw: float
    closes_engineering_net: bool
    cad_ready: bool
    cad_readiness_score: float
    recovery_aggressiveness: float
    direct_heat_recovery_fraction: float
    recovered_electric_power_mw: float
    radiator_temperature_k: float
    radiator_emissivity: float
    topology_packing_factor: float
    inferred_rejected_heat_mw: float
    adjusted_rejected_heat_mw: float
    adjusted_radiator_area_m2: float
    adjusted_wing_span_each_m: float
    machine_length_m: float
    outer_radius_m: float
    source_wing_span_each_m: float
    source_radiator_area_m2: float
    source_blockers: str
    source_design_id: str
    blockers: str
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV-friendly row."""
        return asdict(self)


def evaluate_thermal_recovery(
    envelope: CADEnvelopeV11Result,
    assumptions: Optional[ThermalRecoveryV12Assumptions] = None,
) -> ThermalRecoveryV12Result:
    """Apply one thermal recovery recipe to a CAD envelope row."""
    assumptions = assumptions or ThermalRecoveryV12Assumptions()
    _validate_envelope(envelope)
    unmargin_radiator_area = envelope.radiator_area_m2 / (
        1.0 + assumptions.source_radiator_area_margin
    )
    rejected_heat = _rejected_heat_from_area(
        radiator_area_m2=unmargin_radiator_area,
        temperature_k=assumptions.base_radiator_temperature_k,
        emissivity=assumptions.base_radiator_emissivity,
    )
    recovered_electric_power = rejected_heat * assumptions.direct_heat_recovery_fraction
    adjusted_rejected_heat = rejected_heat * (
        1.0 - assumptions.direct_heat_recovery_fraction
    )
    adjusted_area = _radiator_area_m2(
        rejected_heat_mw=adjusted_rejected_heat,
        temperature_k=assumptions.radiator_temperature_k,
        emissivity=assumptions.radiator_emissivity,
    )
    adjusted_wing_span = adjusted_area / (
        2.0 * envelope.machine_length_m * assumptions.topology_packing_factor
    )
    plant_net = envelope.source_plant_net_power_mw + recovered_electric_power
    closes = plant_net > 0.0
    nonthermal_source_blockers = _nonthermal_source_blockers(envelope.blockers)
    packaging_ok = (
        adjusted_wing_span <= assumptions.max_wing_span_each_m
        and adjusted_area <= assumptions.max_radiator_area_m2
        and envelope.machine_length_m <= assumptions.max_machine_length_m
        and envelope.outer_radius_m <= assumptions.max_outer_radius_m
    )
    cad_ready = (
        envelope.uses_modulus_control
        and closes
        and packaging_ok
        and not nonthermal_source_blockers
    )
    thermal_score = _bounded_score(
        1.0 - adjusted_wing_span / assumptions.max_wing_span_each_m
    )
    power_score = _bounded_score(plant_net / 10.0)
    cad_readiness_score = _bounded_score(
        0.45 * envelope.cad_readiness_score + 0.35 * thermal_score + 0.20 * power_score
    )
    blockers = _blockers(
        closes=closes,
        adjusted_wing_span=adjusted_wing_span,
        adjusted_area=adjusted_area,
        outer_radius=envelope.outer_radius_m,
        machine_length=envelope.machine_length_m,
        nonthermal_source_blockers=nonthermal_source_blockers,
        assumptions=assumptions,
    )

    return ThermalRecoveryV12Result(
        source_cad_readiness_score=envelope.cad_readiness_score,
        source_plant_net_power_mw=envelope.source_plant_net_power_mw,
        plant_net_power_mw=plant_net,
        closes_engineering_net=closes,
        cad_ready=cad_ready,
        cad_readiness_score=cad_readiness_score,
        recovery_aggressiveness=recovery_aggressiveness_score(assumptions),
        direct_heat_recovery_fraction=assumptions.direct_heat_recovery_fraction,
        recovered_electric_power_mw=recovered_electric_power,
        radiator_temperature_k=assumptions.radiator_temperature_k,
        radiator_emissivity=assumptions.radiator_emissivity,
        topology_packing_factor=assumptions.topology_packing_factor,
        inferred_rejected_heat_mw=rejected_heat,
        adjusted_rejected_heat_mw=adjusted_rejected_heat,
        adjusted_radiator_area_m2=adjusted_area,
        adjusted_wing_span_each_m=adjusted_wing_span,
        machine_length_m=envelope.machine_length_m,
        outer_radius_m=envelope.outer_radius_m,
        source_wing_span_each_m=envelope.radiator_wing_span_each_m,
        source_radiator_area_m2=envelope.radiator_area_m2,
        source_blockers=envelope.blockers,
        source_design_id=_source_design_id(envelope),
        blockers=blockers,
        warnings=_warnings(envelope, blockers),
    )


def recovery_aggressiveness_score(assumptions: ThermalRecoveryV12Assumptions) -> float:
    """Score how aggressive a thermal recovery recipe is."""
    direct_score = assumptions.direct_heat_recovery_fraction / 0.01
    temp_score = max((assumptions.radiator_temperature_k - 800.0) / 200.0, 0.0)
    topology_score = max(log2(assumptions.topology_packing_factor), 0.0)
    return direct_score + temp_score + topology_score


def run_thermal_recovery_v12(
    target_screening_net_mw: float = 50.0,
    direct_heat_recovery_fractions: Sequence[
        float
    ] = DEFAULT_DIRECT_HEAT_RECOVERY_FRACTIONS,
    radiator_temperatures_k: Sequence[float] = DEFAULT_RADIATOR_TEMPERATURES_K,
    topology_packing_factors: Sequence[float] = DEFAULT_TOPOLOGY_PACKING_FACTORS,
    limit: Optional[int] = 24,
) -> List[ThermalRecoveryV12Result]:
    """Sweep thermal recovery recipes over v1.1 CAD envelope rows."""
    _require_positive_finite_number("target_screening_net_mw", target_screening_net_mw)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
    _validate_values(
        "direct_heat_recovery_fractions",
        direct_heat_recovery_fractions,
        allow_zero=True,
    )
    _validate_values("radiator_temperatures_k", radiator_temperatures_k)
    _validate_values("topology_packing_factors", topology_packing_factors)

    envelopes = run_cad_envelope_v11(
        target_screening_net_mw=target_screening_net_mw, limit=None
    )
    rows: List[ThermalRecoveryV12Result] = []
    for envelope in envelopes:
        for direct_fraction in direct_heat_recovery_fractions:
            for temperature in radiator_temperatures_k:
                for topology in topology_packing_factors:
                    rows.append(
                        evaluate_thermal_recovery(
                            envelope,
                            ThermalRecoveryV12Assumptions(
                                direct_heat_recovery_fraction=direct_fraction,
                                radiator_temperature_k=temperature,
                                topology_packing_factor=topology,
                            ),
                        )
                    )
    ranked = sorted(
        rows,
        key=_rank_key,
    )
    return ranked[:limit] if limit is not None else ranked


def _rank_key(row: ThermalRecoveryV12Result) -> tuple[bool, float, float]:
    return (not row.cad_ready, row.recovery_aggressiveness, -row.cad_readiness_score)


def _radiator_area_m2(
    rejected_heat_mw: float,
    temperature_k: float,
    emissivity: float,
) -> float:
    heat_w = rejected_heat_mw * 1.0e6
    flux = emissivity * STEFAN_BOLTZMANN_W_M2_K4 * temperature_k**4
    return heat_w / flux


def _rejected_heat_from_area(
    radiator_area_m2: float,
    temperature_k: float,
    emissivity: float,
) -> float:
    flux = emissivity * STEFAN_BOLTZMANN_W_M2_K4 * temperature_k**4
    return radiator_area_m2 * flux / 1.0e6


def _blockers(
    closes: bool,
    adjusted_wing_span: float,
    adjusted_area: float,
    outer_radius: float,
    machine_length: float,
    nonthermal_source_blockers: List[str],
    assumptions: ThermalRecoveryV12Assumptions,
) -> str:
    blockers = list(nonthermal_source_blockers)
    if not closes:
        blockers.append("plant remains net-negative")
    if adjusted_wing_span > assumptions.max_wing_span_each_m:
        blockers.append("radiator wing span remains too large")
    if adjusted_area > assumptions.max_radiator_area_m2:
        blockers.append("radiator area remains too large")
    if machine_length > assumptions.max_machine_length_m:
        blockers.append("machine length remains too large")
    if outer_radius > assumptions.max_outer_radius_m:
        blockers.append("outer radius remains too large")
    return "; ".join(blockers) if blockers else "none"


def _warnings(envelope: CADEnvelopeV11Result, blockers: str) -> str:
    notes = [
        "v1.2 thermal recovery separates direct conversion recovery from radiator packaging",
        f"source v1.1 blockers: {envelope.blockers}",
    ]
    if blockers != "none":
        notes.append(f"v1.2 residual blockers: {blockers}")
    return "; ".join(notes)


def _nonthermal_source_blockers(blockers: str) -> List[str]:
    clearable = {
        "source plant is net-negative",
        "radiator wing span exceeds CAD envelope limit",
        "radiator area exceeds CAD envelope limit",
    }
    if blockers == "none":
        return []
    return [blocker for blocker in blockers.split("; ") if blocker not in clearable]


def _source_design_id(envelope: CADEnvelopeV11Result) -> str:
    return (
        f"r{envelope.plasma_radius_m:.3f}_l{envelope.machine_length_m:.3f}_"
        f"or{envelope.outer_radius_m:.3f}_plant{envelope.source_plant_net_power_mw:.3f}"
    )


def _validate_envelope(envelope: CADEnvelopeV11Result) -> None:
    for field in (
        "cad_readiness_score",
        "source_plant_net_power_mw",
        "radiator_area_m2",
        "radiator_wing_span_each_m",
        "machine_length_m",
        "outer_radius_m",
    ):
        _require_finite_number(field, getattr(envelope, field))
    if not 0.0 <= envelope.cad_readiness_score <= 1.0:
        raise ValueError("cad_readiness_score must be between 0 and 1")
    for field in (
        "radiator_area_m2",
        "radiator_wing_span_each_m",
        "machine_length_m",
        "outer_radius_m",
    ):
        if getattr(envelope, field) <= 0.0:
            raise ValueError(f"{field} must be positive")


def _validate_values(
    name: str,
    values: Sequence[float],
    allow_zero: bool = False,
) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        _require_finite_number(name, value)
        if allow_zero:
            if value < 0.0:
                raise ValueError(f"{name} cannot contain negative values")
        elif value <= 0.0:
            raise ValueError(f"{name} must contain positive values")


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
