"""Plant-net geometry re-score for LunarFire."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite, pi
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


DEFAULT_GEOMETRY_TARGET_MW = 50.0
DEFAULT_GEOMETRY_TEMPERATURES_KEV = (120.0, 150.0, 180.0, 200.0)
DEFAULT_GEOMETRY_DENSITIES_M3 = (2.5e20, 3.5e20, 5.0e20, 7.0e20)
DEFAULT_GEOMETRY_CONFINEMENTS_S = (6.0, 8.0, 12.0, 16.0, 20.0)


@dataclass(frozen=True)
class GeometryEngineeringProfile:
    """Geometry-specific architecture assumptions for plant-net comparison."""

    family: GeometryFamily
    beta: float
    aspect_ratio: float
    shape_factor: float
    direct_conversion_efficiency: float
    thermal_conversion_efficiency: float
    transport_loss_multiplier: float
    z_eff: float
    dd_side_reaction_fraction: float
    current_drive_fraction_of_gross_fusion: float
    direct_conversion_access: float
    stability_confidence: float
    engineering_simplicity: float
    max_field_t: float = 14.0
    max_volume_m3: float = 2000.0
    max_neutron_wall_load_mw_m2: float = 0.25
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.family, GeometryFamily):
            raise ValueError("family must be a GeometryFamily")
        for field in (
            "beta",
            "aspect_ratio",
            "shape_factor",
            "direct_conversion_efficiency",
            "thermal_conversion_efficiency",
            "transport_loss_multiplier",
            "z_eff",
            "dd_side_reaction_fraction",
            "current_drive_fraction_of_gross_fusion",
            "direct_conversion_access",
            "stability_confidence",
            "engineering_simplicity",
            "max_field_t",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            value = getattr(self, field)
            if not isfinite(value):
                raise ValueError(f"{field} must be finite")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if self.aspect_ratio <= 0 or self.shape_factor <= 0:
            raise ValueError("aspect_ratio and shape_factor must be positive")
        for field in (
            "direct_conversion_efficiency",
            "thermal_conversion_efficiency",
            "dd_side_reaction_fraction",
            "current_drive_fraction_of_gross_fusion",
            "direct_conversion_access",
            "stability_confidence",
            "engineering_simplicity",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        if self.transport_loss_multiplier < 0:
            raise ValueError("transport_loss_multiplier cannot be negative")
        if self.z_eff <= 0:
            raise ValueError("z_eff must be positive")
        if self.max_field_t <= 0 or self.max_volume_m3 <= 0:
            raise ValueError("maximum geometry constraints must be positive")
        if self.max_neutron_wall_load_mw_m2 <= 0:
            raise ValueError("max_neutron_wall_load_mw_m2 must be positive")


@dataclass(frozen=True)
class GeometryEngineeringResult:
    """Best plant-net operating point for one geometry family."""

    family: str
    feasible_screening_candidate: bool
    rejection_summary: str
    target_screening_net_mw: float
    plant_net_power_mw: float
    closes_engineering_net: bool
    screening_net_power_mw: float
    engineering_load_mw: float
    gross_fusion_mw: float
    rejected_heat_mw: float
    radiator_area_m2: float
    required_field_t: float
    separatrix_radius_m: float
    length_m: float
    plasma_volume_m3: float
    magnet_mass_proxy_tonnes: float
    neutron_wall_load_mw_m2: float
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    direct_conversion_efficiency: float
    current_drive_fraction_of_gross_fusion: float
    transport_loss_multiplier: float
    profile_notes: str
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


DEFAULT_GEOMETRY_PROFILES = (
    GeometryEngineeringProfile(
        family=GeometryFamily.FRC,
        beta=0.85,
        aspect_ratio=6.0,
        shape_factor=0.85,
        direct_conversion_efficiency=0.78,
        thermal_conversion_efficiency=0.38,
        transport_loss_multiplier=0.15,
        z_eff=1.2,
        dd_side_reaction_fraction=0.03,
        current_drive_fraction_of_gross_fusion=0.020,
        direct_conversion_access=0.95,
        stability_confidence=0.55,
        engineering_simplicity=0.78,
        notes="Lead FRC architecture with compact high-beta plasma and end access.",
    ),
    GeometryEngineeringProfile(
        family=GeometryFamily.MIRROR,
        beta=0.45,
        aspect_ratio=10.0,
        shape_factor=1.0,
        direct_conversion_efficiency=0.82,
        thermal_conversion_efficiency=0.35,
        transport_loss_multiplier=0.30,
        z_eff=1.2,
        dd_side_reaction_fraction=0.03,
        current_drive_fraction_of_gross_fusion=0.015,
        direct_conversion_access=0.88,
        stability_confidence=0.45,
        engineering_simplicity=0.72,
        notes="Linear mirror/nozzle case with better extraction and worse end losses.",
    ),
    GeometryEngineeringProfile(
        family=GeometryFamily.SPHERICAL_TORUS,
        beta=0.28,
        aspect_ratio=1.7,
        shape_factor=2.0,
        direct_conversion_efficiency=0.35,
        thermal_conversion_efficiency=0.42,
        transport_loss_multiplier=0.08,
        z_eff=1.2,
        dd_side_reaction_fraction=0.03,
        current_drive_fraction_of_gross_fusion=0.010,
        direct_conversion_access=0.32,
        stability_confidence=0.70,
        engineering_simplicity=0.42,
        max_field_t=20.0,
        notes="Spherical-torus baseline with better maturity but poorer direct conversion.",
    ),
)


def run_geometry_engineering_rescore(
    target_screening_net_mw: float = DEFAULT_GEOMETRY_TARGET_MW,
    profiles: Sequence[GeometryEngineeringProfile] = DEFAULT_GEOMETRY_PROFILES,
    temperatures_kev: Sequence[float] = DEFAULT_GEOMETRY_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_GEOMETRY_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_GEOMETRY_CONFINEMENTS_S,
) -> List[GeometryEngineeringResult]:
    """Compare geometry families using plant-net engineering accounting."""
    if not isfinite(target_screening_net_mw) or target_screening_net_mw <= 0:
        raise ValueError("target_screening_net_mw must be positive and finite")
    if not profiles:
        raise ValueError("profiles cannot be empty")
    _validate_sweep_values("temperatures_kev", temperatures_kev)
    _validate_sweep_values("densities_m3", densities_m3)
    _validate_sweep_values("confinements_s", confinements_s)

    results: List[GeometryEngineeringResult] = []
    for profile in profiles:
        candidates, rejection_counts = _profile_candidates(
            target_screening_net_mw=target_screening_net_mw,
            profile=profile,
            temperatures_kev=temperatures_kev,
            densities_m3=densities_m3,
            confinements_s=confinements_s,
        )
        if candidates:
            results.append(max(candidates, key=lambda row: row.plant_net_power_mw))
        else:
            results.append(
                _infeasible_result(
                    target_screening_net_mw,
                    profile,
                    rejection_counts,
                )
            )

    return sorted(results, key=_result_sort_key, reverse=True)


def _profile_candidates(
    target_screening_net_mw: float,
    profile: GeometryEngineeringProfile,
    temperatures_kev: Sequence[float],
    densities_m3: Sequence[float],
    confinements_s: Sequence[float],
) -> tuple[List[GeometryEngineeringResult], Counter[str]]:
    rows: List[GeometryEngineeringResult] = []
    rejection_counts: Counter[str] = Counter()
    for temperature_kev in temperatures_kev:
        for density_m3 in densities_m3:
            for confinement_s in confinements_s:
                row, rejection_reason = _candidate(
                    target_screening_net_mw=target_screening_net_mw,
                    profile=profile,
                    temperature_kev=temperature_kev,
                    density_m3=density_m3,
                    confinement_s=confinement_s,
                )
                if row is not None:
                    rows.append(row)
                elif rejection_reason:
                    rejection_counts[rejection_reason] += 1
    return rows, rejection_counts


def _candidate(
    target_screening_net_mw: float,
    profile: GeometryEngineeringProfile,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
) -> tuple[Optional[GeometryEngineeringResult], str]:
    unit_scenario = _scenario(
        target_screening_net_mw=target_screening_net_mw,
        profile=profile,
        temperature_kev=temperature_kev,
        density_m3=density_m3,
        confinement_s=confinement_s,
        volume_m3=1.0,
    )
    unit_trade = evaluate_scenario(unit_scenario)
    volume = unit_trade.required_volume_for_target_m3
    if not isfinite(volume) or volume <= 0:
        return None, "no_positive_screening_net"
    if volume > profile.max_volume_m3:
        return None, "volume_limit"

    scenario = _scenario(
        target_screening_net_mw=target_screening_net_mw,
        profile=profile,
        temperature_kev=temperature_kev,
        density_m3=density_m3,
        confinement_s=confinement_s,
        volume_m3=volume,
    )
    trade = evaluate_scenario(scenario)
    geometry = _geometry_for_volume(profile, volume)
    geometry_score = evaluate_geometry_candidate(geometry, scenario)
    if geometry_score.required_field_t > profile.max_field_t:
        return None, "field_limit"
    if geometry_score.neutron_wall_load_mw_m2 > profile.max_neutron_wall_load_mw_m2:
        return None, "wall_load_limit"

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
        beta=profile.beta,
        required_field_t=geometry_score.required_field_t,
        plasma_volume_m3=volume,
        separatrix_radius_m=geometry.minor_radius_m,
        length_m=geometry.length_m,
        direct_conversion_efficiency=profile.direct_conversion_efficiency,
        thermal_conversion_efficiency=profile.thermal_conversion_efficiency,
        transport_loss_multiplier=profile.transport_loss_multiplier,
        dd_side_reaction_fraction=profile.dd_side_reaction_fraction,
        triple_product_kev_s_m3=trade.triple_product_kev_s_m3,
        gain_proxy=trade.gain_proxy,
        magnetic_energy_mj=geometry_score.magnetic_energy_mj,
        magnet_mass_proxy_tonnes=geometry_score.magnet_mass_proxy_tonnes,
        objective_score=geometry_score.total_score,
        warnings=trade.warnings,
    )
    engineering = evaluate_engineering_net(
        reference=reference,
        assumptions=EngineeringAssumptions(
            current_drive_fraction_of_gross_fusion=(
                profile.current_drive_fraction_of_gross_fusion
            )
        ),
    )
    return GeometryEngineeringResult(
        family=profile.family.value,
        feasible_screening_candidate=True,
        rejection_summary="",
        target_screening_net_mw=target_screening_net_mw,
        plant_net_power_mw=engineering.plant_net_power_mw,
        closes_engineering_net=engineering.closes_engineering_net,
        screening_net_power_mw=engineering.screening_net_power_mw,
        engineering_load_mw=engineering.engineering_load_mw,
        gross_fusion_mw=engineering.gross_fusion_mw,
        rejected_heat_mw=engineering.rejected_heat_mw,
        radiator_area_m2=engineering.radiator_area_m2,
        required_field_t=engineering.required_field_t,
        separatrix_radius_m=engineering.separatrix_radius_m,
        length_m=engineering.length_m,
        plasma_volume_m3=volume,
        magnet_mass_proxy_tonnes=engineering.magnet_mass_proxy_tonnes,
        neutron_wall_load_mw_m2=geometry_score.neutron_wall_load_mw_m2,
        temperature_kev=engineering.temperature_kev,
        ion_density_m3=engineering.ion_density_m3,
        confinement_s=engineering.confinement_s,
        direct_conversion_efficiency=profile.direct_conversion_efficiency,
        current_drive_fraction_of_gross_fusion=(
            profile.current_drive_fraction_of_gross_fusion
        ),
        transport_loss_multiplier=profile.transport_loss_multiplier,
        profile_notes=profile.notes,
        warnings=_warnings(profile, engineering.warnings),
    ), ""


def _infeasible_result(
    target_screening_net_mw: float,
    profile: GeometryEngineeringProfile,
    rejection_counts: Counter[str],
) -> GeometryEngineeringResult:
    rejection_summary = _format_rejection_counts(rejection_counts)
    return GeometryEngineeringResult(
        family=profile.family.value,
        feasible_screening_candidate=False,
        rejection_summary=rejection_summary,
        target_screening_net_mw=target_screening_net_mw,
        plant_net_power_mw=-target_screening_net_mw,
        closes_engineering_net=False,
        screening_net_power_mw=0.0,
        engineering_load_mw=0.0,
        gross_fusion_mw=0.0,
        rejected_heat_mw=0.0,
        radiator_area_m2=0.0,
        required_field_t=0.0,
        separatrix_radius_m=0.0,
        length_m=0.0,
        plasma_volume_m3=0.0,
        magnet_mass_proxy_tonnes=0.0,
        neutron_wall_load_mw_m2=0.0,
        temperature_kev=0.0,
        ion_density_m3=0.0,
        confinement_s=0.0,
        direct_conversion_efficiency=profile.direct_conversion_efficiency,
        current_drive_fraction_of_gross_fusion=(
            profile.current_drive_fraction_of_gross_fusion
        ),
        transport_loss_multiplier=profile.transport_loss_multiplier,
        profile_notes=profile.notes,
        warnings=(
            f"geometry profile: {profile.family.value}; "
            "no feasible screening-net candidate under current assumptions; "
            f"rejections: {rejection_summary}"
        ),
    )


def _result_sort_key(result: GeometryEngineeringResult) -> tuple[bool, float]:
    return (result.feasible_screening_candidate, result.plant_net_power_mw)


def _format_rejection_counts(rejection_counts: Counter[str]) -> str:
    if not rejection_counts:
        return "none"
    return ", ".join(
        f"{reason}={count}" for reason, count in sorted(rejection_counts.items())
    )


def _scenario(
    target_screening_net_mw: float,
    profile: GeometryEngineeringProfile,
    temperature_kev: float,
    density_m3: float,
    confinement_s: float,
    volume_m3: float,
) -> Scenario:
    return Scenario(
        name=(
            f"lunarfire_{profile.family.value}_{target_screening_net_mw:.0f}mw_"
            f"t{temperature_kev:.0f}_n{density_m3:.1e}_tau{confinement_s:.0f}"
        ),
        reaction_key=D_HE3.key,
        temperature_kev=temperature_kev,
        ion_density_m3=density_m3,
        confinement_s=confinement_s,
        plasma_volume_m3=volume_m3,
        direct_conversion_efficiency=profile.direct_conversion_efficiency,
        thermal_conversion_efficiency=profile.thermal_conversion_efficiency,
        target_power_mw=target_screening_net_mw,
        z_eff=profile.z_eff,
        transport_loss_multiplier=profile.transport_loss_multiplier,
        dd_side_reaction_fraction=profile.dd_side_reaction_fraction,
        notes=f"LunarFire geometry engineering re-score for {profile.family.value}.",
    )


def _geometry_for_volume(
    profile: GeometryEngineeringProfile,
    volume_m3: float,
) -> GeometryCandidate:
    if profile.family == GeometryFamily.SPHERICAL_TORUS:
        minor = (
            volume_m3 / (2.0 * pi**2 * profile.aspect_ratio * profile.shape_factor)
        ) ** (1.0 / 3.0)
        major = profile.aspect_ratio * minor
        length = 2.0 * (major + minor)
        major_radius = major
    else:
        minor = (volume_m3 / (pi * profile.aspect_ratio * profile.shape_factor)) ** (
            1.0 / 3.0
        )
        length = profile.aspect_ratio * minor
        major_radius = length / 2.0

    compactness = {
        GeometryFamily.FRC: 0.85,
        GeometryFamily.MIRROR: 0.62,
        GeometryFamily.SPHERICAL_TORUS: 0.55,
    }[profile.family]
    return GeometryCandidate(
        name=f"{profile.family.value}_engineering_candidate",
        family=profile.family,
        volume_m3=volume_m3,
        major_radius_m=major_radius,
        minor_radius_m=minor,
        length_m=length,
        beta_target=profile.beta,
        shape_factor=profile.shape_factor,
        compactness_weight=compactness,
        direct_conversion_access=profile.direct_conversion_access,
        stability_confidence=profile.stability_confidence,
        engineering_simplicity=profile.engineering_simplicity,
        notes=profile.notes,
    )


def _validate_sweep_values(name: str, values: Sequence[float]) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must contain only positive finite values")


def _warnings(profile: GeometryEngineeringProfile, engineering_warnings: str) -> str:
    warnings = [warning for warning in engineering_warnings.split("; ") if warning]
    warnings.append(f"geometry profile: {profile.family.value}")
    warnings.append("plant-net comparison uses rough geometry-specific assumptions")
    return "; ".join(warnings)
