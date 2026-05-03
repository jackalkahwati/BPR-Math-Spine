"""LunarFire minimum viable scale sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Dict, List, Optional, Sequence

from helionis.engineering_net import (
    EngineeringAssumptions,
    EngineeringNetResult,
    evaluate_engineering_net,
)
from helionis.reference_design import ReferenceDesignTarget, solve_all_reference_designs


DEFAULT_TARGETS_MW = (10.0, 25.0, 50.0, 100.0, 250.0)


@dataclass(frozen=True)
class ScaleSweepResult:
    """Best engineering-net result for one target screening-net scale."""

    target_screening_net_mw: float
    plant_net_power_mw: float
    plant_net_margin_mw: float
    plant_net_margin_fraction: float
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
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    current_drive_mw: float
    power_conditioning_loss_mw: float
    conversion_waste_mw: float
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


def run_scale_sweep(
    targets_mw: Sequence[float] = DEFAULT_TARGETS_MW,
    assumptions: Optional[EngineeringAssumptions] = None,
) -> List[ScaleSweepResult]:
    """Evaluate best plant-net candidate at each target screening-net scale."""
    _validate_targets(targets_mw)
    assumptions = assumptions or EngineeringAssumptions()

    results: List[ScaleSweepResult] = []
    for target_mw in targets_mw:
        target = ReferenceDesignTarget(
            name=f"lunarfire_frc_{target_mw:.0f}mw_scale",
            target_screening_net_mw=target_mw,
            max_volume_m3=_scaled_max_volume(target_mw),
            max_neutron_wall_load_mw_m2=0.20,
        )
        references = solve_all_reference_designs(target=target)
        if not references:
            continue
        engineering_rows = [
            evaluate_engineering_net(reference, assumptions=assumptions)
            for reference in references
        ]
        best = max(engineering_rows, key=lambda row: row.plant_net_power_mw)
        results.append(_scale_result(target_mw, best))

    return sorted(results, key=lambda result: result.target_screening_net_mw)


def minimum_viable_scale(
    targets_mw: Sequence[float] = DEFAULT_TARGETS_MW,
    assumptions: Optional[EngineeringAssumptions] = None,
) -> Optional[ScaleSweepResult]:
    """Return the first scale that closes plant-net power, if one exists."""
    for result in run_scale_sweep(targets_mw=targets_mw, assumptions=assumptions):
        if result.closes_engineering_net:
            return result
    return None


def _scale_result(
    target_mw: float,
    result: EngineeringNetResult,
) -> ScaleSweepResult:
    margin = result.plant_net_power_mw
    return ScaleSweepResult(
        target_screening_net_mw=target_mw,
        plant_net_power_mw=result.plant_net_power_mw,
        plant_net_margin_mw=margin,
        plant_net_margin_fraction=margin / target_mw,
        closes_engineering_net=result.closes_engineering_net,
        screening_net_power_mw=result.screening_net_power_mw,
        engineering_load_mw=result.engineering_load_mw,
        gross_fusion_mw=result.gross_fusion_mw,
        rejected_heat_mw=result.rejected_heat_mw,
        radiator_area_m2=result.radiator_area_m2,
        required_field_t=result.required_field_t,
        separatrix_radius_m=result.separatrix_radius_m,
        length_m=result.length_m,
        plasma_volume_m3=_frc_volume(result.separatrix_radius_m, result.length_m),
        magnet_mass_proxy_tonnes=result.magnet_mass_proxy_tonnes,
        temperature_kev=result.temperature_kev,
        ion_density_m3=result.ion_density_m3,
        confinement_s=result.confinement_s,
        current_drive_mw=result.current_drive_mw,
        power_conditioning_loss_mw=result.power_conditioning_loss_mw,
        conversion_waste_mw=result.conversion_waste_mw,
        warnings=result.warnings,
    )


def _validate_targets(targets_mw: Sequence[float]) -> None:
    if not targets_mw:
        raise ValueError("targets_mw cannot be empty")
    for target in targets_mw:
        if not isfinite(target) or target <= 0:
            raise ValueError("targets_mw must contain only positive finite values")


def _scaled_max_volume(target_mw: float) -> float:
    return max(200.0, 20.0 * target_mw)


def _frc_volume(radius_m: float, length_m: float, shape_factor: float = 0.85) -> float:
    return 3.141592653589793 * radius_m**2 * length_m * shape_factor
