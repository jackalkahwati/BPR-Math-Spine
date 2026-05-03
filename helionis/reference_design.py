"""LunarFire 10 MW FRC reference-design sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Dict, List, Optional, Sequence

from helionis.architecture import Scenario, evaluate_scenario
from helionis.geometry import (
    GeometryCandidate,
    GeometryFamily,
    GeometryScore,
    evaluate_geometry_candidate,
)
from helionis.reactions import D_HE3


@dataclass(frozen=True)
class ReferenceDesignTarget:
    """Design constraints for the first LunarFire FRC reference case."""

    name: str = "lunarfire_frc_10mw_v0_1"
    target_screening_net_mw: float = 10.0
    beta: float = 0.85
    aspect_ratio: float = 6.0
    shape_factor: float = 0.85
    direct_conversion_efficiency: float = 0.78
    thermal_conversion_efficiency: float = 0.38
    transport_loss_multiplier: float = 0.15
    z_eff: float = 1.2
    dd_side_reaction_fraction: float = 0.03
    max_field_t: float = 12.0
    max_minor_radius_m: float = 3.0
    max_length_m: float = 20.0
    max_volume_m3: float = 200.0
    max_neutron_wall_load_mw_m2: float = 0.10

    def __post_init__(self) -> None:
        for field in (
            "target_screening_net_mw",
            "beta",
            "aspect_ratio",
            "shape_factor",
            "direct_conversion_efficiency",
            "thermal_conversion_efficiency",
            "transport_loss_multiplier",
            "z_eff",
            "dd_side_reaction_fraction",
            "max_field_t",
            "max_minor_radius_m",
            "max_length_m",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            value = getattr(self, field)
            if not isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.target_screening_net_mw <= 0:
            raise ValueError("target_screening_net_mw must be positive")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if self.aspect_ratio <= 0 or self.shape_factor <= 0:
            raise ValueError("aspect_ratio and shape_factor must be positive")
        for field in ("direct_conversion_efficiency", "thermal_conversion_efficiency"):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        if self.transport_loss_multiplier < 0:
            raise ValueError("transport_loss_multiplier cannot be negative")
        if self.z_eff <= 0:
            raise ValueError("z_eff must be positive")
        if not 0.0 <= self.dd_side_reaction_fraction <= 1.0:
            raise ValueError("dd_side_reaction_fraction must be between 0 and 1")
        for field in (
            "max_field_t",
            "max_minor_radius_m",
            "max_length_m",
            "max_volume_m3",
            "max_neutron_wall_load_mw_m2",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")


@dataclass(frozen=True)
class ReferenceDesignResult:
    """One closed 10 MW FRC design candidate."""

    name: str
    target_screening_net_mw: float
    screening_net_power_mw: float
    gross_fusion_mw: float
    useful_power_mw: float
    bremsstrahlung_loss_mw: float
    transport_loss_mw: float
    neutron_power_mw: float
    neutron_wall_load_mw_m2: float
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    beta: float
    required_field_t: float
    plasma_volume_m3: float
    separatrix_radius_m: float
    length_m: float
    direct_conversion_efficiency: float
    thermal_conversion_efficiency: float
    transport_loss_multiplier: float
    dd_side_reaction_fraction: float
    triple_product_kev_s_m3: float
    gain_proxy: float
    magnetic_energy_mj: float
    magnet_mass_proxy_tonnes: float
    objective_score: float
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


DEFAULT_TEMPERATURES_KEV = (100.0, 120.0, 150.0, 180.0, 200.0)
DEFAULT_DENSITIES_M3 = (2.5e20, 3.5e20, 5.0e20, 7.0e20, 1.0e21)
DEFAULT_CONFINEMENTS_S = (4.0, 6.0, 8.0, 12.0, 16.0, 20.0)


def solve_reference_design(
    target: Optional[ReferenceDesignTarget] = None,
    temperatures_kev: Sequence[float] = DEFAULT_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_CONFINEMENTS_S,
    limit: int = 20,
) -> List[ReferenceDesignResult]:
    """Return feasible 10 MW FRC candidates sorted by compactness objective."""
    target = target or ReferenceDesignTarget()
    if limit <= 0:
        raise ValueError("limit must be positive")
    _validate_sweep_values("temperatures_kev", temperatures_kev)
    _validate_sweep_values("densities_m3", densities_m3)
    _validate_sweep_values("confinements_s", confinements_s)
    candidates: List[ReferenceDesignResult] = []

    for temperature_kev in temperatures_kev:
        for density_m3 in densities_m3:
            for confinement_s in confinements_s:
                result = _candidate_from_operating_point(
                    target=target,
                    temperature_kev=temperature_kev,
                    ion_density_m3=density_m3,
                    confinement_s=confinement_s,
                )
                if result is not None:
                    candidates.append(result)

    return sorted(candidates, key=lambda item: item.objective_score)[:limit]


def solve_all_reference_designs(
    target: Optional[ReferenceDesignTarget] = None,
    temperatures_kev: Sequence[float] = DEFAULT_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_CONFINEMENTS_S,
) -> List[ReferenceDesignResult]:
    """Return all feasible FRC candidates for a target."""
    total_grid = len(temperatures_kev) * len(densities_m3) * len(confinements_s)
    return solve_reference_design(
        target=target,
        temperatures_kev=temperatures_kev,
        densities_m3=densities_m3,
        confinements_s=confinements_s,
        limit=total_grid,
    )


def best_reference_design(
    target: Optional[ReferenceDesignTarget] = None,
) -> ReferenceDesignResult:
    """Return the top-ranked feasible LunarFire 10 MW design candidate."""
    results = solve_reference_design(target=target, limit=1)
    if not results:
        raise ValueError("No feasible LunarFire reference design found")
    return results[0]


def _candidate_from_operating_point(
    target: ReferenceDesignTarget,
    temperature_kev: float,
    ion_density_m3: float,
    confinement_s: float,
) -> Optional[ReferenceDesignResult]:
    unit_scenario = _scenario(
        target=target,
        temperature_kev=temperature_kev,
        ion_density_m3=ion_density_m3,
        confinement_s=confinement_s,
        plasma_volume_m3=1.0,
    )
    unit_trade = evaluate_scenario(unit_scenario)
    if not isfinite(unit_trade.required_volume_for_target_m3):
        return None

    volume = unit_trade.required_volume_for_target_m3
    if volume <= 0 or volume > target.max_volume_m3:
        return None

    scenario = _scenario(
        target=target,
        temperature_kev=temperature_kev,
        ion_density_m3=ion_density_m3,
        confinement_s=confinement_s,
        plasma_volume_m3=volume,
    )
    trade = evaluate_scenario(scenario)
    geometry = _frc_geometry_for_volume(target, volume)
    geometry_score = evaluate_geometry_candidate(geometry, scenario)

    if not _passes_constraints(target, geometry, geometry_score):
        return None

    objective = _objective_score(geometry_score)
    warnings = _warnings(trade.warnings, geometry_score)

    return ReferenceDesignResult(
        name=scenario.name,
        target_screening_net_mw=target.target_screening_net_mw,
        screening_net_power_mw=trade.net_power_mw,
        gross_fusion_mw=trade.fusion_power_mw,
        useful_power_mw=trade.useful_power_mw,
        bremsstrahlung_loss_mw=trade.bremsstrahlung_loss_mw,
        transport_loss_mw=trade.transport_loss_mw,
        neutron_power_mw=trade.neutron_power_mw,
        neutron_wall_load_mw_m2=geometry_score.neutron_wall_load_mw_m2,
        temperature_kev=temperature_kev,
        ion_density_m3=ion_density_m3,
        confinement_s=confinement_s,
        beta=target.beta,
        required_field_t=geometry_score.required_field_t,
        plasma_volume_m3=volume,
        separatrix_radius_m=geometry.minor_radius_m,
        length_m=geometry.length_m,
        direct_conversion_efficiency=target.direct_conversion_efficiency,
        thermal_conversion_efficiency=target.thermal_conversion_efficiency,
        transport_loss_multiplier=target.transport_loss_multiplier,
        dd_side_reaction_fraction=target.dd_side_reaction_fraction,
        triple_product_kev_s_m3=trade.triple_product_kev_s_m3,
        gain_proxy=trade.gain_proxy,
        magnetic_energy_mj=geometry_score.magnetic_energy_mj,
        magnet_mass_proxy_tonnes=geometry_score.magnet_mass_proxy_tonnes,
        objective_score=objective,
        warnings=warnings,
    )


def _validate_sweep_values(name: str, values: Sequence[float]) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must contain only positive finite values")


def _scenario(
    target: ReferenceDesignTarget,
    temperature_kev: float,
    ion_density_m3: float,
    confinement_s: float,
    plasma_volume_m3: float,
) -> Scenario:
    suffix = f"t{temperature_kev:.0f}_n{ion_density_m3:.1e}_tau{confinement_s:.0f}"
    return Scenario(
        name=f"{target.name}_{suffix}",
        reaction_key=D_HE3.key,
        temperature_kev=temperature_kev,
        ion_density_m3=ion_density_m3,
        confinement_s=confinement_s,
        plasma_volume_m3=plasma_volume_m3,
        direct_conversion_efficiency=target.direct_conversion_efficiency,
        thermal_conversion_efficiency=target.thermal_conversion_efficiency,
        target_power_mw=target.target_screening_net_mw,
        z_eff=target.z_eff,
        transport_loss_multiplier=target.transport_loss_multiplier,
        dd_side_reaction_fraction=target.dd_side_reaction_fraction,
        notes="LunarFire 10 MW FRC reference-design sweep candidate.",
    )


def _frc_geometry_for_volume(
    target: ReferenceDesignTarget,
    volume_m3: float,
) -> GeometryCandidate:
    radius = (
        volume_m3
        / (3.141592653589793 * target.aspect_ratio * target.shape_factor)
    ) ** (1.0 / 3.0)
    length = target.aspect_ratio * radius
    return GeometryCandidate(
        name="lunarfire_frc_reference",
        family=GeometryFamily.FRC,
        volume_m3=volume_m3,
        major_radius_m=length / 2.0,
        minor_radius_m=radius,
        length_m=length,
        beta_target=target.beta,
        shape_factor=target.shape_factor,
        compactness_weight=0.85,
        direct_conversion_access=0.95,
        stability_confidence=0.55,
        engineering_simplicity=0.78,
        notes="10 MW LunarFire FRC reference geometry.",
    )


def _passes_constraints(
    target: ReferenceDesignTarget,
    geometry: GeometryCandidate,
    geometry_score: GeometryScore,
) -> bool:
    return (
        geometry_score.required_field_t <= target.max_field_t
        and geometry.minor_radius_m <= target.max_minor_radius_m
        and geometry.length_m <= target.max_length_m
        and geometry_score.neutron_wall_load_mw_m2
        <= target.max_neutron_wall_load_mw_m2
    )


def _objective_score(geometry_score: GeometryScore) -> float:
    return (
        geometry_score.volume_m3
        + 2.0 * geometry_score.required_field_t
        + 0.01 * geometry_score.magnet_mass_proxy_tonnes
        + 20.0 * geometry_score.neutron_wall_load_mw_m2
    )


def _warnings(trade_warnings: str, geometry_score: GeometryScore) -> str:
    warnings = [warning for warning in trade_warnings.split("; ") if warning]
    warnings.append("reference design uses optimistic transport/direct-conversion assumptions")
    if geometry_score.required_field_t > 8.0:
        warnings.append("high-field HTS magnet regime")
    return "; ".join(warnings)
