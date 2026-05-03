"""LunarFire v0.7 same-assumption architecture comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, pi
from numbers import Real
from typing import Dict, List, Optional, Sequence

from helionis.architecture import Scenario, evaluate_scenario
from helionis.engineering_net import EngineeringAssumptions, evaluate_engineering_net
from helionis.geometry import GeometryCandidate, GeometryFamily, evaluate_geometry_candidate
from helionis.mirror_nozzle_v06 import (
    DEFAULT_ALPHA_COLLECTOR_VOLTAGES_KV,
    DEFAULT_PROTON_COLLECTOR_VOLTAGES_KV,
    charged_product_collector_efficiency,
    loss_cone_fraction,
)
from helionis.reference_design import ReferenceDesignResult
from helionis.reactions import D_HE3


DEFAULT_V07_TEMPERATURES_KEV = (180.0, 200.0)
DEFAULT_V07_DENSITIES_M3 = (2.5e20, 3.5e20, 5.0e20)
DEFAULT_V07_CONFINEMENTS_S = (12.0, 16.0, 20.0)
DEFAULT_V07_MIRROR_RATIOS = (3.0, 4.0, 5.0)


@dataclass(frozen=True)
class ArchitectureV07Assumptions:
    """Shared screening assumptions for FRC vs mirror/nozzle comparison."""

    beta_frc: float = 0.85
    beta_mirror: float = 0.45
    frc_aspect_ratio: float = 6.0
    mirror_aspect_ratio: float = 10.0
    frc_shape_factor: float = 0.85
    mirror_shape_factor: float = 1.0
    frc_direct_conversion_efficiency: float = 0.78
    thermal_conversion_efficiency: float = 0.35
    z_eff: float = 1.2
    dd_side_reaction_fraction: float = 0.03
    frc_transport_multiplier: float = 0.15
    turbulent_transport_multiplier: float = 0.06
    loss_cone_transport_scale: float = 3.0
    pitch_angle_scattering_s: float = 80.0
    mirror_stabilization_factor: float = 1.0
    nozzle_expansion_bonus: float = 0.04
    max_direct_conversion_efficiency: float = 0.86
    max_collector_power_density_mw_m2: float = 2.0
    collector_aux_kw_per_m2: float = 5.0
    nozzle_aux_fraction_of_direct_power: float = 0.01
    current_drive_fraction_of_gross_fusion: float = 0.015
    plug_coil_mass_coefficient_tonnes_per_t2: float = 0.20
    max_midplane_field_t: float = 14.0
    max_plug_field_t: float = 65.0
    max_volume_m3: float = 2000.0
    max_neutron_wall_load_mw_m2: float = 0.25

    def __post_init__(self) -> None:
        for field in (
            "beta_frc",
            "beta_mirror",
            "frc_aspect_ratio",
            "mirror_aspect_ratio",
            "frc_shape_factor",
            "mirror_shape_factor",
            "frc_direct_conversion_efficiency",
            "thermal_conversion_efficiency",
            "z_eff",
            "dd_side_reaction_fraction",
            "frc_transport_multiplier",
            "turbulent_transport_multiplier",
            "loss_cone_transport_scale",
            "pitch_angle_scattering_s",
            "mirror_stabilization_factor",
            "nozzle_expansion_bonus",
            "max_direct_conversion_efficiency",
            "max_collector_power_density_mw_m2",
            "collector_aux_kw_per_m2",
            "nozzle_aux_fraction_of_direct_power",
            "current_drive_fraction_of_gross_fusion",
            "plug_coil_mass_coefficient_tonnes_per_t2",
            "max_midplane_field_t",
            "max_plug_field_t",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            _require_finite_number(field, getattr(self, field))
        if not 0.0 < self.beta_frc <= 1.0:
            raise ValueError("beta_frc must be between 0 and 1")
        if not 0.0 < self.beta_mirror <= 1.0:
            raise ValueError("beta_mirror must be between 0 and 1")
        if self.max_direct_conversion_efficiency < self.frc_direct_conversion_efficiency:
            raise ValueError(
                "max_direct_conversion_efficiency must be at least frc_direct_conversion_efficiency"
            )
        for field in (
            "frc_direct_conversion_efficiency",
            "thermal_conversion_efficiency",
            "dd_side_reaction_fraction",
            "nozzle_expansion_bonus",
            "max_direct_conversion_efficiency",
            "current_drive_fraction_of_gross_fusion",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        for field in (
            "frc_aspect_ratio",
            "mirror_aspect_ratio",
            "frc_shape_factor",
            "mirror_shape_factor",
            "z_eff",
            "loss_cone_transport_scale",
            "pitch_angle_scattering_s",
            "mirror_stabilization_factor",
            "max_collector_power_density_mw_m2",
            "max_midplane_field_t",
            "max_plug_field_t",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0.0 <= self.nozzle_aux_fraction_of_direct_power <= 1.0:
            raise ValueError("nozzle_aux_fraction_of_direct_power must be between 0 and 1")
        for field in (
            "frc_transport_multiplier",
            "turbulent_transport_multiplier",
            "collector_aux_kw_per_m2",
            "plug_coil_mass_coefficient_tonnes_per_t2",
        ):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} cannot be negative")


@dataclass(frozen=True)
class ArchitectureV07Result:
    """Best same-assumption architecture row for one geometry family."""

    family: str
    target_screening_net_mw: float
    plant_net_power_mw: float
    closes_engineering_net: bool
    screening_net_power_mw: float
    engineering_load_mw: float
    gross_fusion_mw: float
    rejected_heat_mw: float
    radiator_area_m2: float
    collector_nozzle_load_mw: float
    thermal_conversion_efficiency: float
    direct_conversion_efficiency: float
    transport_loss_multiplier: float
    loss_cone_fraction: float
    pitch_angle_scattering_s: float
    mirror_ratio: float
    midplane_field_t: float
    plug_field_t: float
    separatrix_radius_m: float
    length_m: float
    plasma_volume_m3: float
    magnet_mass_proxy_tonnes: float
    plug_coil_mass_proxy_tonnes: float
    effective_magnet_mass_proxy_tonnes: float
    neutron_wall_load_mw_m2: float
    collector_area_m2: float
    alpha_collector_voltage_kv: float
    proton_collector_voltage_kv: float
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


def mirror_leakage_transport_multiplier(
    mirror_ratio: float,
    confinement_s: float,
    pitch_angle_scattering_s: float,
    assumptions: Optional[ArchitectureV07Assumptions] = None,
) -> float:
    """Estimate mirror transport from loss-cone refilling by scattering."""
    assumptions = assumptions or ArchitectureV07Assumptions()
    _require_positive_finite_number("confinement_s", confinement_s)
    _require_positive_finite_number("pitch_angle_scattering_s", pitch_angle_scattering_s)
    leakage = (
        assumptions.loss_cone_transport_scale
        * loss_cone_fraction(mirror_ratio)
        * (confinement_s / pitch_angle_scattering_s)
        / assumptions.mirror_stabilization_factor
    )
    return assumptions.turbulent_transport_multiplier + leakage


def run_architecture_v07_comparison(
    target_screening_net_mw: float = 50.0,
    assumptions: Optional[ArchitectureV07Assumptions] = None,
    temperatures_kev: Sequence[float] = DEFAULT_V07_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_V07_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_V07_CONFINEMENTS_S,
    mirror_ratios: Sequence[float] = DEFAULT_V07_MIRROR_RATIOS,
    alpha_collector_voltages_kv: Sequence[float] = DEFAULT_ALPHA_COLLECTOR_VOLTAGES_KV,
    proton_collector_voltages_kv: Sequence[float] = DEFAULT_PROTON_COLLECTOR_VOLTAGES_KV,
) -> List[ArchitectureV07Result]:
    """Return best shared-grid/shared-accounting FRC and mirror/nozzle rows."""
    _require_positive_finite_number("target_screening_net_mw", target_screening_net_mw)
    assumptions = assumptions or ArchitectureV07Assumptions()
    _validate_values("temperatures_kev", temperatures_kev)
    _validate_values("densities_m3", densities_m3)
    _validate_values("confinements_s", confinements_s)
    _validate_mirror_ratios(mirror_ratios)
    _validate_values("alpha_collector_voltages_kv", alpha_collector_voltages_kv)
    _validate_values("proton_collector_voltages_kv", proton_collector_voltages_kv)

    frc_rows = _frc_candidates(
        target_screening_net_mw,
        assumptions,
        temperatures_kev,
        densities_m3,
        confinements_s,
    )
    mirror_rows = _mirror_candidates(
        target_screening_net_mw,
        assumptions,
        temperatures_kev,
        densities_m3,
        confinements_s,
        mirror_ratios,
        alpha_collector_voltages_kv,
        proton_collector_voltages_kv,
    )
    results: List[ArchitectureV07Result] = []
    if frc_rows:
        results.append(max(frc_rows, key=lambda row: row.plant_net_power_mw))
    if mirror_rows:
        results.append(max(mirror_rows, key=lambda row: row.plant_net_power_mw))
    return sorted(results, key=lambda row: row.plant_net_power_mw, reverse=True)


def _frc_candidates(
    target_screening_net_mw: float,
    assumptions: ArchitectureV07Assumptions,
    temperatures_kev: Sequence[float],
    densities_m3: Sequence[float],
    confinements_s: Sequence[float],
) -> List[ArchitectureV07Result]:
    rows: List[ArchitectureV07Result] = []
    for temperature_kev in temperatures_kev:
        for density_m3 in densities_m3:
            for confinement_s in confinements_s:
                row = _candidate(
                    family=GeometryFamily.FRC,
                    target_screening_net_mw=target_screening_net_mw,
                    assumptions=assumptions,
                    temperature_kev=temperature_kev,
                    density_m3=density_m3,
                    confinement_s=confinement_s,
                    mirror_ratio=0.0,
                    alpha_collector_voltage_kv=0.0,
                    proton_collector_voltage_kv=0.0,
                    direct_efficiency=assumptions.frc_direct_conversion_efficiency,
                    transport_loss_multiplier=assumptions.frc_transport_multiplier,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _mirror_candidates(
    target_screening_net_mw: float,
    assumptions: ArchitectureV07Assumptions,
    temperatures_kev: Sequence[float],
    densities_m3: Sequence[float],
    confinements_s: Sequence[float],
    mirror_ratios: Sequence[float],
    alpha_collector_voltages_kv: Sequence[float],
    proton_collector_voltages_kv: Sequence[float],
) -> List[ArchitectureV07Result]:
    rows: List[ArchitectureV07Result] = []
    for temperature_kev in temperatures_kev:
        for density_m3 in densities_m3:
            for confinement_s in confinements_s:
                for mirror_ratio in mirror_ratios:
                    transport = mirror_leakage_transport_multiplier(
                        mirror_ratio=mirror_ratio,
                        confinement_s=confinement_s,
                        pitch_angle_scattering_s=assumptions.pitch_angle_scattering_s,
                        assumptions=assumptions,
                    )
                    for alpha_voltage in alpha_collector_voltages_kv:
                        for proton_voltage in proton_collector_voltages_kv:
                            direct_efficiency = min(
                                charged_product_collector_efficiency(
                                    alpha_voltage,
                                    proton_voltage,
                                )
                                + _nozzle_expansion_bonus(mirror_ratio, assumptions),
                                assumptions.max_direct_conversion_efficiency,
                            )
                            row = _candidate(
                                family=GeometryFamily.MIRROR,
                                target_screening_net_mw=target_screening_net_mw,
                                assumptions=assumptions,
                                temperature_kev=temperature_kev,
                                density_m3=density_m3,
                                confinement_s=confinement_s,
                                mirror_ratio=mirror_ratio,
                                alpha_collector_voltage_kv=alpha_voltage,
                                proton_collector_voltage_kv=proton_voltage,
                                direct_efficiency=direct_efficiency,
                                transport_loss_multiplier=transport,
                            )
                            if row is not None:
                                rows.append(row)
    return rows


def _candidate(
    family: GeometryFamily,
    target_screening_net_mw: float,
    assumptions: ArchitectureV07Assumptions,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
    mirror_ratio: float,
    alpha_collector_voltage_kv: float,
    proton_collector_voltage_kv: float,
    direct_efficiency: float,
    transport_loss_multiplier: float,
) -> Optional[ArchitectureV07Result]:
    unit_scenario = _scenario(
        family=family,
        target_screening_net_mw=target_screening_net_mw,
        assumptions=assumptions,
        temperature_kev=temperature_kev,
        density_m3=density_m3,
        confinement_s=confinement_s,
        volume_m3=1.0,
        direct_efficiency=direct_efficiency,
        transport_loss_multiplier=transport_loss_multiplier,
    )
    unit_trade = evaluate_scenario(unit_scenario)
    volume = unit_trade.required_volume_for_target_m3
    if not isfinite(volume) or volume <= 0 or volume > assumptions.max_volume_m3:
        return None

    scenario = _scenario(
        family=family,
        target_screening_net_mw=target_screening_net_mw,
        assumptions=assumptions,
        temperature_kev=temperature_kev,
        density_m3=density_m3,
        confinement_s=confinement_s,
        volume_m3=volume,
        direct_efficiency=direct_efficiency,
        transport_loss_multiplier=transport_loss_multiplier,
    )
    trade = evaluate_scenario(scenario)
    geometry = _geometry_for_volume(family, volume, assumptions)
    geometry_score = evaluate_geometry_candidate(geometry, scenario)
    if geometry_score.required_field_t > assumptions.max_midplane_field_t:
        return None
    if geometry_score.neutron_wall_load_mw_m2 > assumptions.max_neutron_wall_load_mw_m2:
        return None

    plug_field_t = 0.0
    plug_mass = 0.0
    if family == GeometryFamily.MIRROR:
        plug_field_t = geometry_score.required_field_t * mirror_ratio
        if plug_field_t > assumptions.max_plug_field_t:
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
        beta=geometry.beta_target,
        required_field_t=geometry_score.required_field_t,
        plasma_volume_m3=volume,
        separatrix_radius_m=geometry.minor_radius_m,
        length_m=geometry.length_m,
        direct_conversion_efficiency=direct_efficiency,
        thermal_conversion_efficiency=assumptions.thermal_conversion_efficiency,
        transport_loss_multiplier=transport_loss_multiplier,
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
    charged_power_mw = trade.charged_power_mw
    direct_converted_power_mw = charged_power_mw * direct_efficiency
    collector_area = (
        charged_power_mw / assumptions.max_collector_power_density_mw_m2
        if family == GeometryFamily.MIRROR
        else 0.0
    )
    collector_nozzle_load_mw = (
        collector_area * assumptions.collector_aux_kw_per_m2 / 1000.0
        + direct_converted_power_mw * assumptions.nozzle_aux_fraction_of_direct_power
        if family == GeometryFamily.MIRROR
        else 0.0
    )
    plant_net_power_mw = engineering.plant_net_power_mw - collector_nozzle_load_mw
    engineering_load_mw = engineering.engineering_load_mw + collector_nozzle_load_mw
    rejected_heat_mw = engineering.rejected_heat_mw + collector_nozzle_load_mw
    return ArchitectureV07Result(
        family="mirror_nozzle" if family == GeometryFamily.MIRROR else family.value,
        target_screening_net_mw=target_screening_net_mw,
        plant_net_power_mw=plant_net_power_mw,
        closes_engineering_net=plant_net_power_mw > 0.0,
        screening_net_power_mw=engineering.screening_net_power_mw,
        engineering_load_mw=engineering_load_mw,
        gross_fusion_mw=engineering.gross_fusion_mw,
        rejected_heat_mw=rejected_heat_mw,
        radiator_area_m2=_radiator_area_m2(rejected_heat_mw),
        collector_nozzle_load_mw=collector_nozzle_load_mw,
        thermal_conversion_efficiency=assumptions.thermal_conversion_efficiency,
        direct_conversion_efficiency=direct_efficiency,
        transport_loss_multiplier=transport_loss_multiplier,
        loss_cone_fraction=loss_cone_fraction(mirror_ratio)
        if family == GeometryFamily.MIRROR
        else 0.0,
        pitch_angle_scattering_s=assumptions.pitch_angle_scattering_s
        if family == GeometryFamily.MIRROR
        else 0.0,
        mirror_ratio=mirror_ratio,
        midplane_field_t=engineering.required_field_t,
        plug_field_t=plug_field_t,
        separatrix_radius_m=engineering.separatrix_radius_m,
        length_m=engineering.length_m,
        plasma_volume_m3=volume,
        magnet_mass_proxy_tonnes=geometry_score.magnet_mass_proxy_tonnes,
        plug_coil_mass_proxy_tonnes=plug_mass,
        effective_magnet_mass_proxy_tonnes=engineering.magnet_mass_proxy_tonnes,
        neutron_wall_load_mw_m2=geometry_score.neutron_wall_load_mw_m2,
        collector_area_m2=collector_area,
        alpha_collector_voltage_kv=alpha_collector_voltage_kv,
        proton_collector_voltage_kv=proton_collector_voltage_kv,
        temperature_kev=engineering.temperature_kev,
        ion_density_m3=engineering.ion_density_m3,
        confinement_s=engineering.confinement_s,
        warnings=_warnings(family, engineering.warnings, collector_nozzle_load_mw),
    )


def _scenario(
    family: GeometryFamily,
    target_screening_net_mw: float,
    assumptions: ArchitectureV07Assumptions,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
    volume_m3: float,
    direct_efficiency: float,
    transport_loss_multiplier: float,
) -> Scenario:
    return Scenario(
        name=(
            f"lunarfire_v07_{family.value}_{target_screening_net_mw:.0f}mw_"
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
        notes="LunarFire v0.7 same-assumption architecture comparison.",
    )


def _geometry_for_volume(
    family: GeometryFamily,
    volume_m3: float,
    assumptions: ArchitectureV07Assumptions,
) -> GeometryCandidate:
    if family == GeometryFamily.FRC:
        radius = (
            volume_m3 / (pi * assumptions.frc_aspect_ratio * assumptions.frc_shape_factor)
        ) ** (1.0 / 3.0)
        length = assumptions.frc_aspect_ratio * radius
        return GeometryCandidate(
            name="lunarfire_v07_frc_candidate",
            family=GeometryFamily.FRC,
            volume_m3=volume_m3,
            major_radius_m=length / 2.0,
            minor_radius_m=radius,
            length_m=length,
            beta_target=assumptions.beta_frc,
            shape_factor=assumptions.frc_shape_factor,
            compactness_weight=0.85,
            direct_conversion_access=0.95,
            stability_confidence=0.55,
            engineering_simplicity=0.78,
            notes="v0.7 FRC same-assumption baseline.",
        )

    radius = (
        volume_m3
        / (pi * assumptions.mirror_aspect_ratio * assumptions.mirror_shape_factor)
    ) ** (1.0 / 3.0)
    length = assumptions.mirror_aspect_ratio * radius
    return GeometryCandidate(
        name="lunarfire_v07_mirror_candidate",
        family=GeometryFamily.MIRROR,
        volume_m3=volume_m3,
        major_radius_m=length / 2.0,
        minor_radius_m=radius,
        length_m=length,
        beta_target=assumptions.beta_mirror,
        shape_factor=assumptions.mirror_shape_factor,
        compactness_weight=0.62,
        direct_conversion_access=0.88,
        stability_confidence=0.45,
        engineering_simplicity=0.72,
        notes="v0.7 mirror/nozzle same-assumption candidate.",
    )


def _nozzle_expansion_bonus(
    mirror_ratio: float,
    assumptions: ArchitectureV07Assumptions,
) -> float:
    return assumptions.nozzle_expansion_bonus * min((mirror_ratio - 1.0) / 4.0, 1.0)


def _plug_coil_mass_proxy_tonnes(
    midplane_field_t: float,
    plug_field_t: float,
    assumptions: ArchitectureV07Assumptions,
) -> float:
    field_lift_t = max(plug_field_t - midplane_field_t, 0.0)
    return assumptions.plug_coil_mass_coefficient_tonnes_per_t2 * field_lift_t**2


def _radiator_area_m2(
    rejected_heat_mw: float,
    radiator_temperature_k: float = 800.0,
    emissivity: float = 0.85,
) -> float:
    stefan_boltzmann_w_m2_k4 = 5.670374419e-8
    heat_w = rejected_heat_mw * 1.0e6
    flux_w_m2 = emissivity * stefan_boltzmann_w_m2_k4 * radiator_temperature_k**4
    return heat_w / flux_w_m2


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
        raise ValueError(f"{name} must be positive")


def _warnings(
    family: GeometryFamily,
    engineering_warnings: str,
    collector_nozzle_load_mw: float,
) -> str:
    warnings = [warning for warning in engineering_warnings.split("; ") if warning]
    warnings.append("v0.7 shared-grid/shared-accounting architecture comparison")
    if family == GeometryFamily.MIRROR:
        warnings.append("mirror leakage uses pitch-angle-scattering proxy")
        warnings.append(
            f"collector/nozzle auxiliary load included: {collector_nozzle_load_mw:.1f} MW"
        )
    else:
        warnings.append("FRC row uses same target/grid/engineering path")
    return "; ".join(warnings)
