"""LunarFire v0.8 mirror/nozzle margin recovery sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, log2
from numbers import Real
from typing import Dict, List, Optional, Sequence

from helionis.architecture_comparison_v07 import (
    ArchitectureV07Assumptions,
    ArchitectureV07Result,
    _candidate,
    mirror_leakage_transport_multiplier,
)
from helionis.geometry import GeometryFamily
from helionis.mirror_nozzle_v06 import (
    D_HE3_ALPHA_ENERGY_MEV,
    D_HE3_PROTON_ENERGY_MEV,
)
from helionis.reactions import D_HE3


DEFAULT_V08_TEMPERATURES_KEV = (200.0,)
DEFAULT_V08_DENSITIES_M3 = (2.5e20, 3.5e20)
DEFAULT_V08_CONFINEMENTS_S = (16.0, 20.0)
DEFAULT_V08_MIRROR_RATIOS = (4.0, 5.0)
DEFAULT_V08_PITCH_SCATTERING_S = (80.0, 160.0)
DEFAULT_V08_STABILIZATION_FACTORS = (1.0, 2.0)
DEFAULT_V08_DIRECT_CAPS = (0.86, 0.88, 0.90)
DEFAULT_V08_COLLECTOR_BASE_EFFICIENCIES = (0.58, 0.60, 0.62)
DEFAULT_V08_COLLECTOR_MATCH_BONUSES = (0.22, 0.30)
DEFAULT_V08_PLUG_COEFFS = (0.20, 0.10)
DEFAULT_V08_COLLECTOR_AUX_KW_M2 = (5.0, 2.0)
DEFAULT_V08_NOZZLE_AUX_FRACTIONS = (0.010,)


@dataclass(frozen=True)
class MarginRecoveryV08Assumptions:
    """One margin-recovery recipe for mirror/nozzle."""

    pitch_angle_scattering_s: float = 80.0
    mirror_stabilization_factor: float = 1.0
    direct_conversion_cap: float = 0.86
    collector_base_efficiency: float = 0.58
    collector_match_bonus: float = 0.22
    plug_coil_mass_coefficient_tonnes_per_t2: float = 0.20
    collector_aux_kw_per_m2: float = 5.0
    nozzle_aux_fraction_of_direct_power: float = 0.010

    def __post_init__(self) -> None:
        for field in (
            "pitch_angle_scattering_s",
            "mirror_stabilization_factor",
            "direct_conversion_cap",
            "collector_base_efficiency",
            "collector_match_bonus",
            "plug_coil_mass_coefficient_tonnes_per_t2",
            "collector_aux_kw_per_m2",
            "nozzle_aux_fraction_of_direct_power",
        ):
            _require_finite_number(field, getattr(self, field))
        for field in ("pitch_angle_scattering_s", "mirror_stabilization_factor"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        for field in (
            "direct_conversion_cap",
            "collector_base_efficiency",
            "collector_match_bonus",
            "nozzle_aux_fraction_of_direct_power",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        if self.direct_conversion_cap < self.collector_base_efficiency:
            raise ValueError("direct_conversion_cap must be at least collector_base_efficiency")
        for field in (
            "plug_coil_mass_coefficient_tonnes_per_t2",
            "collector_aux_kw_per_m2",
        ):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} cannot be negative")


@dataclass(frozen=True)
class MarginRecoveryV08Result:
    """One mirror/nozzle recovery sensitivity row."""

    target_screening_net_mw: float
    plant_net_power_mw: float
    closes_engineering_net: bool
    plant_net_margin_mw: float
    aggressiveness_score: float
    pitch_angle_scattering_s: float
    mirror_stabilization_factor: float
    direct_conversion_cap: float
    collector_base_efficiency: float
    collector_match_bonus: float
    uncapped_direct_conversion_efficiency: float
    is_direct_conversion_cap_limited: bool
    direct_conversion_efficiency: float
    plug_coil_mass_coefficient_tonnes_per_t2: float
    collector_aux_kw_per_m2: float
    nozzle_aux_fraction_of_direct_power: float
    mirror_ratio: float
    transport_loss_multiplier: float
    plug_field_t: float
    collector_nozzle_load_mw: float
    engineering_load_mw: float
    gross_fusion_mw: float
    rejected_heat_mw: float
    radiator_area_m2: float
    collector_area_m2: float
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


def recovery_aggressiveness_score(assumptions: MarginRecoveryV08Assumptions) -> float:
    """Score how aggressive a recovery recipe is relative to v0.7 nominal."""
    scattering_score = max(log2(assumptions.pitch_angle_scattering_s / 80.0), 0.0)
    stabilization_score = max(log2(assumptions.mirror_stabilization_factor), 0.0)
    direct_score = max((assumptions.direct_conversion_cap - 0.86) / 0.02, 0.0)
    base_score = max((assumptions.collector_base_efficiency - 0.58) / 0.02, 0.0)
    match_score = max((assumptions.collector_match_bonus - 0.22) / 0.04, 0.0)
    plug_score = max((0.20 - assumptions.plug_coil_mass_coefficient_tonnes_per_t2) / 0.05, 0.0)
    collector_score = max((5.0 - assumptions.collector_aux_kw_per_m2) / 1.5, 0.0)
    nozzle_score = max((0.010 - assumptions.nozzle_aux_fraction_of_direct_power) / 0.0025, 0.0)
    return (
        scattering_score
        + stabilization_score
        + direct_score
        + base_score
        + match_score
        + plug_score
        + collector_score
        + nozzle_score
    )


def run_margin_recovery_v08_sweep(
    target_screening_net_mw: float = 50.0,
    temperatures_kev: Sequence[float] = DEFAULT_V08_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_V08_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_V08_CONFINEMENTS_S,
    mirror_ratios: Sequence[float] = DEFAULT_V08_MIRROR_RATIOS,
    pitch_angle_scattering_s: Sequence[float] = DEFAULT_V08_PITCH_SCATTERING_S,
    mirror_stabilization_factors: Sequence[float] = DEFAULT_V08_STABILIZATION_FACTORS,
    direct_conversion_caps: Sequence[float] = DEFAULT_V08_DIRECT_CAPS,
    collector_base_efficiencies: Sequence[float] = DEFAULT_V08_COLLECTOR_BASE_EFFICIENCIES,
    collector_match_bonuses: Sequence[float] = DEFAULT_V08_COLLECTOR_MATCH_BONUSES,
    plug_coil_mass_coefficients: Sequence[float] = DEFAULT_V08_PLUG_COEFFS,
    collector_aux_kw_per_m2: Sequence[float] = DEFAULT_V08_COLLECTOR_AUX_KW_M2,
    nozzle_aux_fractions: Sequence[float] = DEFAULT_V08_NOZZLE_AUX_FRACTIONS,
    limit: Optional[int] = 24,
) -> List[MarginRecoveryV08Result]:
    """Sweep recovery knobs and rank likely closing recipes first."""
    _require_positive_finite_number("target_screening_net_mw", target_screening_net_mw)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
    _validate_values("temperatures_kev", temperatures_kev)
    _validate_values("densities_m3", densities_m3)
    _validate_values("confinements_s", confinements_s)
    _validate_mirror_ratios(mirror_ratios)
    _validate_values("pitch_angle_scattering_s", pitch_angle_scattering_s)
    _validate_values("mirror_stabilization_factors", mirror_stabilization_factors)
    _validate_values("direct_conversion_caps", direct_conversion_caps)
    _validate_values("collector_base_efficiencies", collector_base_efficiencies)
    _validate_values("collector_match_bonuses", collector_match_bonuses)
    _validate_values("plug_coil_mass_coefficients", plug_coil_mass_coefficients, allow_zero=True)
    _validate_values("collector_aux_kw_per_m2", collector_aux_kw_per_m2, allow_zero=True)
    _validate_values("nozzle_aux_fractions", nozzle_aux_fractions, allow_zero=True)

    rows: List[MarginRecoveryV08Result] = []
    for scattering_s in pitch_angle_scattering_s:
        for stabilization in mirror_stabilization_factors:
            for direct_cap in direct_conversion_caps:
                for collector_base in collector_base_efficiencies:
                    for collector_match in collector_match_bonuses:
                        for plug_coeff in plug_coil_mass_coefficients:
                            for collector_aux in collector_aux_kw_per_m2:
                                for nozzle_aux in nozzle_aux_fractions:
                                    recipe = MarginRecoveryV08Assumptions(
                                        pitch_angle_scattering_s=scattering_s,
                                        mirror_stabilization_factor=stabilization,
                                        direct_conversion_cap=direct_cap,
                                        collector_base_efficiency=collector_base,
                                        collector_match_bonus=collector_match,
                                        plug_coil_mass_coefficient_tonnes_per_t2=plug_coeff,
                                        collector_aux_kw_per_m2=collector_aux,
                                        nozzle_aux_fraction_of_direct_power=nozzle_aux,
                                    )
                                    rows.extend(
                                        _recipe_rows(
                                            target_screening_net_mw,
                                            recipe,
                                            temperatures_kev,
                                            densities_m3,
                                            confinements_s,
                                            mirror_ratios,
                                        )
                                    )

    ranked = sorted(rows, key=_rank_key)
    return ranked[:limit] if limit is not None else ranked


def minimum_recovery_recipe(
    target_screening_net_mw: float = 50.0,
) -> Optional[MarginRecoveryV08Result]:
    """Return the least aggressive closing recipe, if one exists."""
    rows = run_margin_recovery_v08_sweep(
        target_screening_net_mw=target_screening_net_mw,
        limit=None,
    )
    closing_rows = [row for row in rows if row.closes_engineering_net]
    if not closing_rows:
        return None
    return min(
        closing_rows,
        key=lambda row: (row.aggressiveness_score, -row.plant_net_power_mw),
    )


def _recipe_rows(
    target_screening_net_mw: float,
    recipe: MarginRecoveryV08Assumptions,
    temperatures_kev: Sequence[float],
    densities_m3: Sequence[float],
    confinements_s: Sequence[float],
    mirror_ratios: Sequence[float],
) -> List[MarginRecoveryV08Result]:
    rows: List[MarginRecoveryV08Result] = []
    arch_assumptions = ArchitectureV07Assumptions(
        pitch_angle_scattering_s=recipe.pitch_angle_scattering_s,
        mirror_stabilization_factor=recipe.mirror_stabilization_factor,
        max_direct_conversion_efficiency=recipe.direct_conversion_cap,
        plug_coil_mass_coefficient_tonnes_per_t2=(
            recipe.plug_coil_mass_coefficient_tonnes_per_t2
        ),
        collector_aux_kw_per_m2=recipe.collector_aux_kw_per_m2,
        nozzle_aux_fraction_of_direct_power=recipe.nozzle_aux_fraction_of_direct_power,
    )
    alpha_voltage_kv = (D_HE3_ALPHA_ENERGY_MEV / 2.0) * 1000.0
    proton_voltage_kv = D_HE3_PROTON_ENERGY_MEV * 1000.0
    for temperature_kev in temperatures_kev:
        for density_m3 in densities_m3:
            for confinement_s in confinements_s:
                for mirror_ratio in mirror_ratios:
                    transport = mirror_leakage_transport_multiplier(
                        mirror_ratio=mirror_ratio,
                        confinement_s=confinement_s,
                        pitch_angle_scattering_s=recipe.pitch_angle_scattering_s,
                        assumptions=arch_assumptions,
                    )
                    uncapped_direct_efficiency = (
                        _raw_collector_efficiency(
                            alpha_voltage_kv,
                            proton_voltage_kv,
                            recipe,
                        )
                        + _nozzle_expansion_bonus(mirror_ratio, arch_assumptions)
                    )
                    direct_efficiency = min(
                        uncapped_direct_efficiency,
                        recipe.direct_conversion_cap,
                    )
                    architecture_row = _candidate(
                        family=GeometryFamily.MIRROR,
                        target_screening_net_mw=target_screening_net_mw,
                        assumptions=arch_assumptions,
                        temperature_kev=temperature_kev,
                        density_m3=density_m3,
                        confinement_s=confinement_s,
                        mirror_ratio=mirror_ratio,
                        alpha_collector_voltage_kv=alpha_voltage_kv,
                        proton_collector_voltage_kv=proton_voltage_kv,
                        direct_efficiency=direct_efficiency,
                        transport_loss_multiplier=transport,
                    )
                    if architecture_row is not None:
                        rows.append(
                            _result(
                                target_screening_net_mw,
                                recipe,
                                architecture_row,
                                uncapped_direct_efficiency,
                            )
                        )
    return rows


def _result(
    target_screening_net_mw: float,
    recipe: MarginRecoveryV08Assumptions,
    row: ArchitectureV07Result,
    uncapped_direct_efficiency: float,
) -> MarginRecoveryV08Result:
    return MarginRecoveryV08Result(
        target_screening_net_mw=target_screening_net_mw,
        plant_net_power_mw=row.plant_net_power_mw,
        closes_engineering_net=row.closes_engineering_net,
        plant_net_margin_mw=row.plant_net_power_mw,
        aggressiveness_score=recovery_aggressiveness_score(recipe),
        pitch_angle_scattering_s=recipe.pitch_angle_scattering_s,
        mirror_stabilization_factor=recipe.mirror_stabilization_factor,
        direct_conversion_cap=recipe.direct_conversion_cap,
        collector_base_efficiency=recipe.collector_base_efficiency,
        collector_match_bonus=recipe.collector_match_bonus,
        uncapped_direct_conversion_efficiency=uncapped_direct_efficiency,
        is_direct_conversion_cap_limited=(
            row.direct_conversion_efficiency < uncapped_direct_efficiency
        ),
        direct_conversion_efficiency=row.direct_conversion_efficiency,
        plug_coil_mass_coefficient_tonnes_per_t2=(
            recipe.plug_coil_mass_coefficient_tonnes_per_t2
        ),
        collector_aux_kw_per_m2=recipe.collector_aux_kw_per_m2,
        nozzle_aux_fraction_of_direct_power=recipe.nozzle_aux_fraction_of_direct_power,
        mirror_ratio=row.mirror_ratio,
        transport_loss_multiplier=row.transport_loss_multiplier,
        plug_field_t=row.plug_field_t,
        collector_nozzle_load_mw=row.collector_nozzle_load_mw,
        engineering_load_mw=row.engineering_load_mw,
        gross_fusion_mw=row.gross_fusion_mw,
        rejected_heat_mw=row.rejected_heat_mw,
        radiator_area_m2=row.radiator_area_m2,
        collector_area_m2=row.collector_area_m2,
        temperature_kev=row.temperature_kev,
        ion_density_m3=row.ion_density_m3,
        confinement_s=row.confinement_s,
        warnings=_warnings(row),
    )


def _rank_key(row: MarginRecoveryV08Result) -> tuple[bool, float, float]:
    return (not row.closes_engineering_net, row.aggressiveness_score, -row.plant_net_power_mw)


def _nozzle_expansion_bonus(
    mirror_ratio: float,
    assumptions: ArchitectureV07Assumptions,
) -> float:
    return assumptions.nozzle_expansion_bonus * min((mirror_ratio - 1.0) / 4.0, 1.0)


def _raw_collector_efficiency(
    alpha_collector_voltage_kv: float,
    proton_collector_voltage_kv: float,
    recipe: MarginRecoveryV08Assumptions,
) -> float:
    alpha_ideal_kv = (D_HE3_ALPHA_ENERGY_MEV / 2.0) * 1000.0
    proton_ideal_kv = D_HE3_PROTON_ENERGY_MEV * 1000.0
    alpha_weight = D_HE3_ALPHA_ENERGY_MEV / D_HE3.q_mev
    proton_weight = D_HE3_PROTON_ENERGY_MEV / D_HE3.q_mev
    weighted_match = (
        alpha_weight * _voltage_match(alpha_collector_voltage_kv, alpha_ideal_kv)
        + proton_weight * _voltage_match(proton_collector_voltage_kv, proton_ideal_kv)
    )
    return recipe.collector_base_efficiency + recipe.collector_match_bonus * weighted_match


def _voltage_match(actual_kv: float, ideal_kv: float) -> float:
    return max(0.0, 1.0 - abs(actual_kv - ideal_kv) / ideal_kv)


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
            if value < 0:
                raise ValueError(f"{name} cannot contain negative values")
        elif value <= 0:
            raise ValueError(f"{name} must contain only positive finite values")


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


def _warnings(row: ArchitectureV07Result) -> str:
    warnings = [warning for warning in row.warnings.split("; ") if warning]
    warnings.append("v0.8 margin-recovery sensitivity recipe")
    if row.closes_engineering_net:
        warnings.append("closes under recovery assumptions")
    return "; ".join(warnings)
