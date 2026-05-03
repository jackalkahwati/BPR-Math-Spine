"""Dedicated mirror/nozzle model for LunarFire v0.5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, pi
from numbers import Real
from typing import Dict, List, Optional, Sequence

from helionis.architecture import Scenario, evaluate_scenario
from helionis.engineering_net import EngineeringAssumptions, evaluate_engineering_net
from helionis.geometry import (
    GeometryCandidate,
    GeometryFamily,
    evaluate_geometry_candidate,
)
from helionis.reference_design import ReferenceDesignResult
from helionis.reactions import D_HE3


DEFAULT_MIRROR_TEMPERATURES_KEV = (150.0, 180.0, 200.0)
DEFAULT_MIRROR_DENSITIES_M3 = (2.5e20, 3.5e20, 5.0e20, 7.0e20)
DEFAULT_MIRROR_CONFINEMENTS_S = (8.0, 12.0, 16.0, 20.0)
DEFAULT_MIRROR_RATIOS = (2.5, 3.0, 4.0, 5.0)
DEFAULT_COLLECTOR_VOLTAGES_KV = (750.0, 1000.0, 1500.0, 2000.0, 2500.0)


@dataclass(frozen=True)
class MirrorNozzleAssumptions:
    """Architecture assumptions for explicit mirror/nozzle screening."""

    beta: float = 0.45
    aspect_ratio: float = 10.0
    shape_factor: float = 1.0
    thermal_conversion_efficiency: float = 0.35
    z_eff: float = 1.2
    dd_side_reaction_fraction: float = 0.03
    turbulent_transport_multiplier: float = 0.06
    end_loss_floor: float = 0.04
    end_loss_coefficient: float = 1.6
    collector_voltage_target_kv: float = 1500.0
    base_direct_conversion_efficiency: float = 0.64
    voltage_match_bonus: float = 0.18
    nozzle_expansion_bonus: float = 0.06
    max_direct_conversion_efficiency: float = 0.88
    current_drive_fraction_of_gross_fusion: float = 0.015
    plug_coil_mass_coefficient_tonnes_per_t2: float = 0.20
    max_midplane_field_t: float = 14.0
    max_plug_field_t: float = 65.0
    max_volume_m3: float = 2000.0
    max_neutron_wall_load_mw_m2: float = 0.25

    def __post_init__(self) -> None:
        for field in (
            "beta",
            "aspect_ratio",
            "shape_factor",
            "thermal_conversion_efficiency",
            "z_eff",
            "dd_side_reaction_fraction",
            "turbulent_transport_multiplier",
            "end_loss_floor",
            "end_loss_coefficient",
            "collector_voltage_target_kv",
            "base_direct_conversion_efficiency",
            "voltage_match_bonus",
            "nozzle_expansion_bonus",
            "max_direct_conversion_efficiency",
            "current_drive_fraction_of_gross_fusion",
            "plug_coil_mass_coefficient_tonnes_per_t2",
            "max_midplane_field_t",
            "max_plug_field_t",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            value = getattr(self, field)
            _require_finite_number(field, value)
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if self.aspect_ratio <= 0 or self.shape_factor <= 0:
            raise ValueError("aspect_ratio and shape_factor must be positive")
        for field in (
            "thermal_conversion_efficiency",
            "dd_side_reaction_fraction",
            "base_direct_conversion_efficiency",
            "voltage_match_bonus",
            "nozzle_expansion_bonus",
            "max_direct_conversion_efficiency",
            "current_drive_fraction_of_gross_fusion",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        if self.max_direct_conversion_efficiency < self.base_direct_conversion_efficiency:
            raise ValueError(
                "max_direct_conversion_efficiency must be at least base efficiency"
            )
        for field in (
            "z_eff",
            "collector_voltage_target_kv",
            "max_midplane_field_t",
            "max_plug_field_t",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.turbulent_transport_multiplier < 0:
            raise ValueError("turbulent_transport_multiplier cannot be negative")
        if self.end_loss_floor < 0 or self.end_loss_coefficient < 0:
            raise ValueError("end-loss terms cannot be negative")
        if self.plug_coil_mass_coefficient_tonnes_per_t2 < 0:
            raise ValueError("plug_coil_mass_coefficient_tonnes_per_t2 cannot be negative")


@dataclass(frozen=True)
class MirrorNozzleResult:
    """One explicit mirror/nozzle operating point."""

    target_screening_net_mw: float
    plant_net_power_mw: float
    closes_engineering_net: bool
    screening_net_power_mw: float
    engineering_load_mw: float
    gross_fusion_mw: float
    rejected_heat_mw: float
    radiator_area_m2: float
    midplane_field_t: float
    mirror_ratio: float
    plug_field_t: float
    max_plug_field_t: float
    collector_voltage_kv: float
    direct_conversion_efficiency: float
    end_loss_multiplier: float
    transport_loss_multiplier: float
    separatrix_radius_m: float
    length_m: float
    plasma_volume_m3: float
    magnet_mass_proxy_tonnes: float
    plug_coil_mass_proxy_tonnes: float
    effective_magnet_mass_proxy_tonnes: float
    neutron_wall_load_mw_m2: float
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


def mirror_end_loss_multiplier(
    mirror_ratio: float,
    assumptions: Optional[MirrorNozzleAssumptions] = None,
) -> float:
    """Estimate axial end-loss multiplier as a simple mirror-ratio penalty."""
    assumptions = assumptions or MirrorNozzleAssumptions()
    if not isfinite(mirror_ratio) or mirror_ratio <= 1.0:
        raise ValueError("mirror_ratio must be finite and greater than 1")
    return assumptions.end_loss_floor + assumptions.end_loss_coefficient / mirror_ratio


def direct_conversion_efficiency_for_collector(
    collector_voltage_kv: float,
    assumptions: Optional[MirrorNozzleAssumptions] = None,
) -> float:
    """Estimate direct-conversion efficiency from collector voltage matching."""
    assumptions = assumptions or MirrorNozzleAssumptions()
    if not isfinite(collector_voltage_kv) or collector_voltage_kv <= 0:
        raise ValueError("collector_voltage_kv must be positive and finite")

    target = assumptions.collector_voltage_target_kv
    match = max(0.0, 1.0 - abs(collector_voltage_kv - target) / target)
    efficiency = assumptions.base_direct_conversion_efficiency + (
        assumptions.voltage_match_bonus * match
    )
    return min(efficiency, assumptions.max_direct_conversion_efficiency)


def run_mirror_nozzle_sweep(
    target_screening_net_mw: float = 50.0,
    assumptions: Optional[MirrorNozzleAssumptions] = None,
    temperatures_kev: Sequence[float] = DEFAULT_MIRROR_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_MIRROR_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_MIRROR_CONFINEMENTS_S,
    mirror_ratios: Sequence[float] = DEFAULT_MIRROR_RATIOS,
    collector_voltages_kv: Sequence[float] = DEFAULT_COLLECTOR_VOLTAGES_KV,
    limit: Optional[int] = 12,
) -> List[MirrorNozzleResult]:
    """Sweep explicit mirror/nozzle assumptions and rank by plant-net power."""
    if not isfinite(target_screening_net_mw) or target_screening_net_mw <= 0:
        raise ValueError("target_screening_net_mw must be positive and finite")
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
    assumptions = assumptions or MirrorNozzleAssumptions()
    _validate_values("temperatures_kev", temperatures_kev)
    _validate_values("densities_m3", densities_m3)
    _validate_values("confinements_s", confinements_s)
    _validate_values("collector_voltages_kv", collector_voltages_kv)
    _validate_mirror_ratios(mirror_ratios)

    rows: List[MirrorNozzleResult] = []
    for temperature_kev in temperatures_kev:
        for density_m3 in densities_m3:
            for confinement_s in confinements_s:
                for mirror_ratio in mirror_ratios:
                    for collector_voltage_kv in collector_voltages_kv:
                        row = _candidate(
                            target_screening_net_mw=target_screening_net_mw,
                            assumptions=assumptions,
                            temperature_kev=temperature_kev,
                            density_m3=density_m3,
                            confinement_s=confinement_s,
                            mirror_ratio=mirror_ratio,
                            collector_voltage_kv=collector_voltage_kv,
                        )
                        if row is not None:
                            rows.append(row)

    ranked = sorted(rows, key=lambda row: row.plant_net_power_mw, reverse=True)
    return ranked[:limit] if limit is not None else ranked


def _candidate(
    target_screening_net_mw: float,
    assumptions: MirrorNozzleAssumptions,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
    mirror_ratio: float,
    collector_voltage_kv: float,
) -> Optional[MirrorNozzleResult]:
    end_loss = mirror_end_loss_multiplier(mirror_ratio, assumptions)
    transport = assumptions.turbulent_transport_multiplier + end_loss
    direct_efficiency = _direct_efficiency_with_nozzle(
        collector_voltage_kv,
        mirror_ratio,
        assumptions,
    )
    unit_scenario = _scenario(
        target_screening_net_mw=target_screening_net_mw,
        assumptions=assumptions,
        temperature_kev=temperature_kev,
        density_m3=density_m3,
        confinement_s=confinement_s,
        volume_m3=1.0,
        direct_efficiency=direct_efficiency,
        transport_loss_multiplier=transport,
    )
    unit_trade = evaluate_scenario(unit_scenario)
    volume = unit_trade.required_volume_for_target_m3
    if not isfinite(volume) or volume <= 0 or volume > assumptions.max_volume_m3:
        return None

    scenario = _scenario(
        target_screening_net_mw=target_screening_net_mw,
        assumptions=assumptions,
        temperature_kev=temperature_kev,
        density_m3=density_m3,
        confinement_s=confinement_s,
        volume_m3=volume,
        direct_efficiency=direct_efficiency,
        transport_loss_multiplier=transport,
    )
    trade = evaluate_scenario(scenario)
    geometry = _mirror_geometry_for_volume(volume, assumptions)
    geometry_score = evaluate_geometry_candidate(geometry, scenario)
    if geometry_score.required_field_t > assumptions.max_midplane_field_t:
        return None

    plug_field_t = geometry_score.required_field_t * mirror_ratio
    if plug_field_t > assumptions.max_plug_field_t:
        return None
    if geometry_score.neutron_wall_load_mw_m2 > assumptions.max_neutron_wall_load_mw_m2:
        return None
    plug_coil_mass_proxy_tonnes = _plug_coil_mass_proxy_tonnes(
        midplane_field_t=geometry_score.required_field_t,
        plug_field_t=plug_field_t,
        assumptions=assumptions,
    )
    effective_magnet_mass_proxy_tonnes = (
        geometry_score.magnet_mass_proxy_tonnes + plug_coil_mass_proxy_tonnes
    )

    reference = ReferenceDesignResult(
        name=scenario.name,
        target_screening_net_mw=target_screening_net_mw,
        screening_net_power_mw=trade.net_power_mw,
        gross_fusion_mw=trade.fusion_power_mw,
        useful_power_mw=trade.useful_power_mw,
        bremsstrahlung_loss_mw=trade.bremsstrahlung_loss_mw,
        transport_loss_mw=trade.transport_loss_mw,
        neutron_power_mw=trade.neutron_power_mw,
        neutron_wall_load_mw_m2=geometry_score.neutron_wall_load_mw_m2,
        temperature_kev=temperature_kev,
        ion_density_m3=density_m3,
        confinement_s=confinement_s,
        beta=assumptions.beta,
        required_field_t=geometry_score.required_field_t,
        plasma_volume_m3=volume,
        separatrix_radius_m=geometry.minor_radius_m,
        length_m=geometry.length_m,
        direct_conversion_efficiency=direct_efficiency,
        thermal_conversion_efficiency=assumptions.thermal_conversion_efficiency,
        transport_loss_multiplier=transport,
        dd_side_reaction_fraction=assumptions.dd_side_reaction_fraction,
        triple_product_kev_s_m3=trade.triple_product_kev_s_m3,
        gain_proxy=trade.gain_proxy,
        magnetic_energy_mj=geometry_score.magnetic_energy_mj,
        magnet_mass_proxy_tonnes=effective_magnet_mass_proxy_tonnes,
        objective_score=geometry_score.total_score,
        warnings=trade.warnings,
    )
    engineering = evaluate_engineering_net(
        reference=reference,
        assumptions=EngineeringAssumptions(
            current_drive_fraction_of_gross_fusion=(
                assumptions.current_drive_fraction_of_gross_fusion
            )
        ),
    )
    return MirrorNozzleResult(
        target_screening_net_mw=target_screening_net_mw,
        plant_net_power_mw=engineering.plant_net_power_mw,
        closes_engineering_net=engineering.closes_engineering_net,
        screening_net_power_mw=engineering.screening_net_power_mw,
        engineering_load_mw=engineering.engineering_load_mw,
        gross_fusion_mw=engineering.gross_fusion_mw,
        rejected_heat_mw=engineering.rejected_heat_mw,
        radiator_area_m2=engineering.radiator_area_m2,
        midplane_field_t=engineering.required_field_t,
        mirror_ratio=mirror_ratio,
        plug_field_t=plug_field_t,
        max_plug_field_t=assumptions.max_plug_field_t,
        collector_voltage_kv=collector_voltage_kv,
        direct_conversion_efficiency=direct_efficiency,
        end_loss_multiplier=end_loss,
        transport_loss_multiplier=transport,
        separatrix_radius_m=engineering.separatrix_radius_m,
        length_m=engineering.length_m,
        plasma_volume_m3=volume,
        magnet_mass_proxy_tonnes=geometry_score.magnet_mass_proxy_tonnes,
        plug_coil_mass_proxy_tonnes=plug_coil_mass_proxy_tonnes,
        effective_magnet_mass_proxy_tonnes=engineering.magnet_mass_proxy_tonnes,
        neutron_wall_load_mw_m2=geometry_score.neutron_wall_load_mw_m2,
        temperature_kev=engineering.temperature_kev,
        ion_density_m3=engineering.ion_density_m3,
        confinement_s=engineering.confinement_s,
        warnings=_warnings(engineering.warnings),
    )


def _direct_efficiency_with_nozzle(
    collector_voltage_kv: float,
    mirror_ratio: float,
    assumptions: MirrorNozzleAssumptions,
) -> float:
    collector_efficiency = direct_conversion_efficiency_for_collector(
        collector_voltage_kv,
        assumptions,
    )
    expansion_score = min((mirror_ratio - 1.0) / 4.0, 1.0)
    return min(
        collector_efficiency + assumptions.nozzle_expansion_bonus * expansion_score,
        assumptions.max_direct_conversion_efficiency,
    )


def _plug_coil_mass_proxy_tonnes(
    midplane_field_t: float,
    plug_field_t: float,
    assumptions: MirrorNozzleAssumptions,
) -> float:
    field_lift_t = max(plug_field_t - midplane_field_t, 0.0)
    return assumptions.plug_coil_mass_coefficient_tonnes_per_t2 * field_lift_t**2


def _scenario(
    target_screening_net_mw: float,
    assumptions: MirrorNozzleAssumptions,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
    volume_m3: float,
    direct_efficiency: float,
    transport_loss_multiplier: float,
) -> Scenario:
    return Scenario(
        name=(
            "lunarfire_mirror_nozzle_"
            f"{target_screening_net_mw:.0f}mw_"
            f"t{temperature_kev:.0f}_n{density_m3:.1e}_tau{confinement_s:.0f}"
        ),
        reaction_key=D_HE3.key,
        temperature_kev=temperature_kev,
        ion_density_m3=density_m3,
        confinement_s=confinement_s,
        plasma_volume_m3=volume_m3,
        direct_conversion_efficiency=direct_efficiency,
        thermal_conversion_efficiency=assumptions.thermal_conversion_efficiency,
        target_power_mw=target_screening_net_mw,
        z_eff=assumptions.z_eff,
        transport_loss_multiplier=transport_loss_multiplier,
        dd_side_reaction_fraction=assumptions.dd_side_reaction_fraction,
        notes="LunarFire v0.5 explicit mirror/nozzle candidate.",
    )


def _mirror_geometry_for_volume(
    volume_m3: float,
    assumptions: MirrorNozzleAssumptions,
) -> GeometryCandidate:
    radius = (
        volume_m3 / (pi * assumptions.aspect_ratio * assumptions.shape_factor)
    ) ** (1.0 / 3.0)
    length = assumptions.aspect_ratio * radius
    return GeometryCandidate(
        name="lunarfire_mirror_nozzle_candidate",
        family=GeometryFamily.MIRROR,
        volume_m3=volume_m3,
        major_radius_m=length / 2.0,
        minor_radius_m=radius,
        length_m=length,
        beta_target=assumptions.beta,
        shape_factor=assumptions.shape_factor,
        compactness_weight=0.62,
        direct_conversion_access=0.88,
        stability_confidence=0.45,
        engineering_simplicity=0.72,
        notes="Explicit v0.5 mirror/nozzle geometry.",
    )


def _validate_values(name: str, values: Sequence[float]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must contain only positive finite values")


def _validate_mirror_ratios(values: Sequence[float]) -> None:
    if len(values) == 0:
        raise ValueError("mirror_ratios cannot be empty")
    for value in values:
        if not isfinite(value) or value <= 1.0:
            raise ValueError("mirror_ratios must be finite and greater than 1")


def _require_finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _warnings(engineering_warnings: str) -> str:
    warnings = [warning for warning in engineering_warnings.split("; ") if warning]
    warnings.append("mirror/nozzle v0.5 uses explicit mirror-ratio end-loss proxy")
    warnings.append("collector voltage is a staged direct-conversion proxy")
    return "; ".join(warnings)
