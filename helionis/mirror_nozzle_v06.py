"""Mirror/nozzle v0.6 model with loss-cone and staged collector proxies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, pi, sqrt
from numbers import Real
from typing import Dict, List, Optional, Sequence

from helionis.architecture import Scenario, evaluate_scenario
from helionis.engineering_net import EngineeringAssumptions, evaluate_engineering_net
from helionis.geometry import GeometryCandidate, GeometryFamily, evaluate_geometry_candidate
from helionis.reference_design import ReferenceDesignResult
from helionis.reactions import D_HE3


D_HE3_ALPHA_ENERGY_MEV = 3.673
D_HE3_PROTON_ENERGY_MEV = 14.680
DEFAULT_V06_TEMPERATURES_KEV = (180.0, 200.0)
DEFAULT_V06_DENSITIES_M3 = (2.5e20, 3.5e20, 5.0e20)
DEFAULT_V06_CONFINEMENTS_S = (12.0, 16.0, 20.0)
DEFAULT_V06_MIRROR_RATIOS = (3.0, 4.0, 5.0)
DEFAULT_ALPHA_COLLECTOR_VOLTAGES_KV = (1200.0, 1800.0, 2400.0)
DEFAULT_PROTON_COLLECTOR_VOLTAGES_KV = (10000.0, 12000.0, 15000.0)


@dataclass(frozen=True)
class MirrorNozzleV06Assumptions:
    """Architecture assumptions for v0.6 mirror/nozzle screening."""

    beta: float = 0.45
    aspect_ratio: float = 10.0
    shape_factor: float = 1.0
    thermal_conversion_efficiency: float = 0.35
    z_eff: float = 1.2
    dd_side_reaction_fraction: float = 0.03
    turbulent_transport_multiplier: float = 0.06
    loss_cone_transport_scale: float = 3.0
    collector_base_efficiency: float = 0.58
    collector_match_bonus: float = 0.22
    nozzle_expansion_bonus: float = 0.04
    max_direct_conversion_efficiency: float = 0.86
    max_collector_power_density_mw_m2: float = 2.0
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
            "loss_cone_transport_scale",
            "collector_base_efficiency",
            "collector_match_bonus",
            "nozzle_expansion_bonus",
            "max_direct_conversion_efficiency",
            "max_collector_power_density_mw_m2",
            "current_drive_fraction_of_gross_fusion",
            "plug_coil_mass_coefficient_tonnes_per_t2",
            "max_midplane_field_t",
            "max_plug_field_t",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            _require_finite_number(field, getattr(self, field))
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if self.aspect_ratio <= 0 or self.shape_factor <= 0:
            raise ValueError("aspect_ratio and shape_factor must be positive")
        for field in (
            "thermal_conversion_efficiency",
            "dd_side_reaction_fraction",
            "collector_base_efficiency",
            "collector_match_bonus",
            "nozzle_expansion_bonus",
            "max_direct_conversion_efficiency",
            "current_drive_fraction_of_gross_fusion",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        if self.max_direct_conversion_efficiency < self.collector_base_efficiency:
            raise ValueError("max_direct_conversion_efficiency must exceed base")
        for field in (
            "z_eff",
            "loss_cone_transport_scale",
            "max_collector_power_density_mw_m2",
            "max_midplane_field_t",
            "max_plug_field_t",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.turbulent_transport_multiplier < 0:
            raise ValueError("turbulent_transport_multiplier cannot be negative")
        if self.plug_coil_mass_coefficient_tonnes_per_t2 < 0:
            raise ValueError("plug_coil_mass_coefficient_tonnes_per_t2 cannot be negative")


@dataclass(frozen=True)
class MirrorNozzleV06Result:
    """One v0.6 mirror/nozzle operating point."""

    target_screening_net_mw: float
    plant_net_power_mw: float
    closes_engineering_net: bool
    screening_net_power_mw: float
    engineering_load_mw: float
    gross_fusion_mw: float
    charged_power_mw: float
    direct_converted_power_mw: float
    rejected_heat_mw: float
    radiator_area_m2: float
    midplane_field_t: float
    mirror_ratio: float
    loss_cone_fraction: float
    plug_field_t: float
    max_plug_field_t: float
    alpha_collector_voltage_kv: float
    proton_collector_voltage_kv: float
    direct_conversion_efficiency: float
    loss_cone_transport_scale: float
    transport_loss_multiplier: float
    collector_area_m2: float
    assumed_collector_power_density_mw_m2: float
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


def loss_cone_fraction(mirror_ratio: float) -> float:
    """Return isotropic two-ended loss-cone fraction for a mirror ratio."""
    if not isfinite(mirror_ratio) or mirror_ratio <= 1.0:
        raise ValueError("mirror_ratio must be finite and greater than 1")
    root = sqrt(1.0 - 1.0 / mirror_ratio)
    return 1.0 / (mirror_ratio * (1.0 + root))


def charged_product_collector_efficiency(
    alpha_collector_voltage_kv: float,
    proton_collector_voltage_kv: float,
    assumptions: Optional[MirrorNozzleV06Assumptions] = None,
) -> float:
    """Estimate staged alpha/proton direct-conversion efficiency."""
    assumptions = assumptions or MirrorNozzleV06Assumptions()
    _require_positive_finite_number(
        "alpha_collector_voltage_kv",
        alpha_collector_voltage_kv,
    )
    _require_positive_finite_number(
        "proton_collector_voltage_kv",
        proton_collector_voltage_kv,
    )

    alpha_ideal_kv = (D_HE3_ALPHA_ENERGY_MEV / 2.0) * 1000.0
    proton_ideal_kv = D_HE3_PROTON_ENERGY_MEV * 1000.0
    alpha_match = _voltage_match(alpha_collector_voltage_kv, alpha_ideal_kv)
    proton_match = _voltage_match(proton_collector_voltage_kv, proton_ideal_kv)
    alpha_weight = D_HE3_ALPHA_ENERGY_MEV / D_HE3.q_mev
    proton_weight = D_HE3_PROTON_ENERGY_MEV / D_HE3.q_mev
    weighted_match = alpha_weight * alpha_match + proton_weight * proton_match
    return min(
        assumptions.collector_base_efficiency
        + assumptions.collector_match_bonus * weighted_match,
        assumptions.max_direct_conversion_efficiency,
    )


def run_mirror_nozzle_v06_sweep(
    target_screening_net_mw: float = 50.0,
    assumptions: Optional[MirrorNozzleV06Assumptions] = None,
    temperatures_kev: Sequence[float] = DEFAULT_V06_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_V06_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_V06_CONFINEMENTS_S,
    mirror_ratios: Sequence[float] = DEFAULT_V06_MIRROR_RATIOS,
    alpha_collector_voltages_kv: Sequence[float] = DEFAULT_ALPHA_COLLECTOR_VOLTAGES_KV,
    proton_collector_voltages_kv: Sequence[float] = DEFAULT_PROTON_COLLECTOR_VOLTAGES_KV,
    limit: Optional[int] = 12,
) -> List[MirrorNozzleV06Result]:
    """Sweep v0.6 mirror/nozzle assumptions and rank by plant-net power."""
    _require_positive_finite_number("target_screening_net_mw", target_screening_net_mw)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
    assumptions = assumptions or MirrorNozzleV06Assumptions()
    _validate_values("temperatures_kev", temperatures_kev)
    _validate_values("densities_m3", densities_m3)
    _validate_values("confinements_s", confinements_s)
    _validate_mirror_ratios(mirror_ratios)
    _validate_values("alpha_collector_voltages_kv", alpha_collector_voltages_kv)
    _validate_values("proton_collector_voltages_kv", proton_collector_voltages_kv)

    rows: List[MirrorNozzleV06Result] = []
    for temperature_kev in temperatures_kev:
        for density_m3 in densities_m3:
            for confinement_s in confinements_s:
                for mirror_ratio in mirror_ratios:
                    for alpha_voltage in alpha_collector_voltages_kv:
                        for proton_voltage in proton_collector_voltages_kv:
                            row = _candidate(
                                target_screening_net_mw=target_screening_net_mw,
                                assumptions=assumptions,
                                temperature_kev=temperature_kev,
                                density_m3=density_m3,
                                confinement_s=confinement_s,
                                mirror_ratio=mirror_ratio,
                                alpha_collector_voltage_kv=alpha_voltage,
                                proton_collector_voltage_kv=proton_voltage,
                            )
                            if row is not None:
                                rows.append(row)

    ranked = sorted(rows, key=lambda row: row.plant_net_power_mw, reverse=True)
    return ranked[:limit] if limit is not None else ranked


def _candidate(
    target_screening_net_mw: float,
    assumptions: MirrorNozzleV06Assumptions,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
    mirror_ratio: float,
    alpha_collector_voltage_kv: float,
    proton_collector_voltage_kv: float,
) -> Optional[MirrorNozzleV06Result]:
    loss_fraction = loss_cone_fraction(mirror_ratio)
    transport = (
        assumptions.turbulent_transport_multiplier
        + assumptions.loss_cone_transport_scale * loss_fraction
    )
    direct_efficiency = min(
        charged_product_collector_efficiency(
            alpha_collector_voltage_kv,
            proton_collector_voltage_kv,
            assumptions,
        )
        + _nozzle_expansion_bonus(mirror_ratio, assumptions),
        assumptions.max_direct_conversion_efficiency,
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

    plug_mass = _plug_coil_mass_proxy_tonnes(
        midplane_field_t=geometry_score.required_field_t,
        plug_field_t=plug_field_t,
        assumptions=assumptions,
    )
    effective_magnet_mass = geometry_score.magnet_mass_proxy_tonnes + plug_mass
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
        magnet_mass_proxy_tonnes=effective_magnet_mass,
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
    direct_power = trade.charged_power_mw * direct_efficiency
    collector_area = (
        trade.charged_power_mw / assumptions.max_collector_power_density_mw_m2
    )
    return MirrorNozzleV06Result(
        target_screening_net_mw=target_screening_net_mw,
        plant_net_power_mw=engineering.plant_net_power_mw,
        closes_engineering_net=engineering.closes_engineering_net,
        screening_net_power_mw=engineering.screening_net_power_mw,
        engineering_load_mw=engineering.engineering_load_mw,
        gross_fusion_mw=engineering.gross_fusion_mw,
        charged_power_mw=trade.charged_power_mw,
        direct_converted_power_mw=direct_power,
        rejected_heat_mw=engineering.rejected_heat_mw,
        radiator_area_m2=engineering.radiator_area_m2,
        midplane_field_t=engineering.required_field_t,
        mirror_ratio=mirror_ratio,
        loss_cone_fraction=loss_fraction,
        plug_field_t=plug_field_t,
        max_plug_field_t=assumptions.max_plug_field_t,
        alpha_collector_voltage_kv=alpha_collector_voltage_kv,
        proton_collector_voltage_kv=proton_collector_voltage_kv,
        direct_conversion_efficiency=direct_efficiency,
        loss_cone_transport_scale=assumptions.loss_cone_transport_scale,
        transport_loss_multiplier=transport,
        collector_area_m2=collector_area,
        assumed_collector_power_density_mw_m2=(
            assumptions.max_collector_power_density_mw_m2
        ),
        separatrix_radius_m=engineering.separatrix_radius_m,
        length_m=engineering.length_m,
        plasma_volume_m3=volume,
        magnet_mass_proxy_tonnes=geometry_score.magnet_mass_proxy_tonnes,
        plug_coil_mass_proxy_tonnes=plug_mass,
        effective_magnet_mass_proxy_tonnes=engineering.magnet_mass_proxy_tonnes,
        neutron_wall_load_mw_m2=geometry_score.neutron_wall_load_mw_m2,
        temperature_kev=engineering.temperature_kev,
        ion_density_m3=engineering.ion_density_m3,
        confinement_s=engineering.confinement_s,
        warnings=_warnings(engineering.warnings),
    )


def _scenario(
    target_screening_net_mw: float,
    assumptions: MirrorNozzleV06Assumptions,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
    volume_m3: float,
    direct_efficiency: float,
    transport_loss_multiplier: float,
) -> Scenario:
    return Scenario(
        name=(
            "lunarfire_mirror_nozzle_v06_"
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
        notes="LunarFire v0.6 mirror/nozzle candidate.",
    )


def _mirror_geometry_for_volume(
    volume_m3: float,
    assumptions: MirrorNozzleV06Assumptions,
) -> GeometryCandidate:
    radius = (
        volume_m3 / (pi * assumptions.aspect_ratio * assumptions.shape_factor)
    ) ** (1.0 / 3.0)
    length = assumptions.aspect_ratio * radius
    return GeometryCandidate(
        name="lunarfire_mirror_nozzle_v06_candidate",
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
        notes="Explicit v0.6 mirror/nozzle geometry.",
    )


def _voltage_match(actual_kv: float, ideal_kv: float) -> float:
    return max(0.0, 1.0 - abs(actual_kv - ideal_kv) / ideal_kv)


def _nozzle_expansion_bonus(
    mirror_ratio: float,
    assumptions: MirrorNozzleV06Assumptions,
) -> float:
    return assumptions.nozzle_expansion_bonus * min((mirror_ratio - 1.0) / 4.0, 1.0)


def _plug_coil_mass_proxy_tonnes(
    midplane_field_t: float,
    plug_field_t: float,
    assumptions: MirrorNozzleV06Assumptions,
) -> float:
    field_lift_t = max(plug_field_t - midplane_field_t, 0.0)
    return assumptions.plug_coil_mass_coefficient_tonnes_per_t2 * field_lift_t**2


def _validate_values(name: str, values: Sequence[float]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        _require_positive_finite_number(name, value)


def _validate_mirror_ratios(values: Sequence[float]) -> None:
    if len(values) == 0:
        raise ValueError("mirror_ratios cannot be empty")
    for value in values:
        _require_finite_number("mirror_ratio", value)
        if value <= 1.0:
            raise ValueError("mirror_ratios must be finite and greater than 1")


def _require_finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _require_positive_finite_number(name: str, value: object) -> None:
    _require_finite_number(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _warnings(engineering_warnings: str) -> str:
    warnings = [warning for warning in engineering_warnings.split("; ") if warning]
    warnings.append("mirror/nozzle v0.6 uses isotropic loss-cone proxy")
    warnings.append("direct conversion uses staged alpha/proton collector proxy")
    return "; ".join(warnings)
