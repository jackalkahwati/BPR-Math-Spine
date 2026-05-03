"""LunarFire engineering net-power budget."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Dict, List, Optional

from helionis.reference_design import (
    ReferenceDesignResult,
    best_reference_design,
    solve_reference_design,
)


STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


@dataclass(frozen=True)
class EngineeringAssumptions:
    """Subsystem assumptions that convert screening-net to plant-net power."""

    current_drive_fraction_of_gross_fusion: float = 0.02
    formation_energy_mj: float = 50.0
    pulse_repetition_hz: float = 0.03
    cryogenic_cold_load_kw_per_magnet_tonne: float = 0.25
    cryogenic_wallplug_w_per_cold_w: float = 25.0
    power_conditioning_loss_fraction: float = 0.04
    thermal_rejection_parasitic_fraction: float = 0.005
    fixed_balance_of_plant_mw: float = 1.5
    radiator_temperature_k: float = 800.0
    radiator_emissivity: float = 0.85

    def __post_init__(self) -> None:
        for field in (
            "current_drive_fraction_of_gross_fusion",
            "formation_energy_mj",
            "pulse_repetition_hz",
            "cryogenic_cold_load_kw_per_magnet_tonne",
            "cryogenic_wallplug_w_per_cold_w",
            "power_conditioning_loss_fraction",
            "thermal_rejection_parasitic_fraction",
            "fixed_balance_of_plant_mw",
            "radiator_temperature_k",
            "radiator_emissivity",
        ):
            value = getattr(self, field)
            if not isfinite(value):
                raise ValueError(f"{field} must be finite")

        for field in (
            "current_drive_fraction_of_gross_fusion",
            "power_conditioning_loss_fraction",
            "thermal_rejection_parasitic_fraction",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")
        if not 0.0 < self.radiator_emissivity <= 1.0:
            raise ValueError("radiator_emissivity must be between 0 and 1")

        for field in (
            "formation_energy_mj",
            "pulse_repetition_hz",
            "cryogenic_cold_load_kw_per_magnet_tonne",
            "cryogenic_wallplug_w_per_cold_w",
            "radiator_temperature_k",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")

        if self.fixed_balance_of_plant_mw < 0:
            raise ValueError("fixed_balance_of_plant_mw cannot be negative")


@dataclass(frozen=True)
class EngineeringNetResult:
    """Engineering net-power budget for one LunarFire reference candidate."""

    reference_name: str
    screening_net_power_mw: float
    plant_net_power_mw: float
    engineering_load_mw: float
    current_drive_mw: float
    formation_average_mw: float
    cryogenic_wallplug_mw: float
    power_conditioning_loss_mw: float
    thermal_rejection_parasitic_mw: float
    fixed_balance_of_plant_mw: float
    conversion_waste_mw: float
    rejected_heat_mw: float
    radiator_area_m2: float
    gross_fusion_mw: float
    useful_power_mw: float
    bremsstrahlung_loss_mw: float
    transport_loss_mw: float
    magnetic_energy_mj: float
    magnet_mass_proxy_tonnes: float
    required_field_t: float
    separatrix_radius_m: float
    length_m: float
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    closes_engineering_net: bool
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


def evaluate_engineering_net(
    reference: Optional[ReferenceDesignResult] = None,
    assumptions: Optional[EngineeringAssumptions] = None,
) -> EngineeringNetResult:
    """Convert one screening-net reference result into plant-net power."""
    reference = reference or best_reference_design()
    assumptions = assumptions or EngineeringAssumptions()

    current_drive_mw = (
        reference.gross_fusion_mw
        * assumptions.current_drive_fraction_of_gross_fusion
    )
    formation_average_mw = (
        assumptions.formation_energy_mj * assumptions.pulse_repetition_hz
    )
    cryogenic_wallplug_mw = (
        reference.magnet_mass_proxy_tonnes
        * assumptions.cryogenic_cold_load_kw_per_magnet_tonne
        * assumptions.cryogenic_wallplug_w_per_cold_w
        / 1000.0
    )
    power_conditioning_loss_mw = (
        reference.useful_power_mw * assumptions.power_conditioning_loss_fraction
    )
    conversion_waste_mw = reference.gross_fusion_mw - reference.useful_power_mw
    rejected_heat_mw = (
        conversion_waste_mw
        + reference.bremsstrahlung_loss_mw
        + reference.transport_loss_mw
        + power_conditioning_loss_mw
        + current_drive_mw
        + formation_average_mw
        + cryogenic_wallplug_mw
        + assumptions.fixed_balance_of_plant_mw
    )
    thermal_rejection_parasitic_mw = (
        rejected_heat_mw * assumptions.thermal_rejection_parasitic_fraction
    )
    engineering_load_mw = (
        current_drive_mw
        + formation_average_mw
        + cryogenic_wallplug_mw
        + power_conditioning_loss_mw
        + thermal_rejection_parasitic_mw
        + assumptions.fixed_balance_of_plant_mw
    )
    plant_net_power_mw = reference.screening_net_power_mw - engineering_load_mw
    radiator_area_m2 = _radiator_area_m2(
        rejected_heat_mw,
        assumptions.radiator_temperature_k,
        assumptions.radiator_emissivity,
    )

    warnings = _warnings(reference, plant_net_power_mw)

    return EngineeringNetResult(
        reference_name=reference.name,
        screening_net_power_mw=reference.screening_net_power_mw,
        plant_net_power_mw=plant_net_power_mw,
        engineering_load_mw=engineering_load_mw,
        current_drive_mw=current_drive_mw,
        formation_average_mw=formation_average_mw,
        cryogenic_wallplug_mw=cryogenic_wallplug_mw,
        power_conditioning_loss_mw=power_conditioning_loss_mw,
        thermal_rejection_parasitic_mw=thermal_rejection_parasitic_mw,
        fixed_balance_of_plant_mw=assumptions.fixed_balance_of_plant_mw,
        conversion_waste_mw=conversion_waste_mw,
        rejected_heat_mw=rejected_heat_mw,
        radiator_area_m2=radiator_area_m2,
        gross_fusion_mw=reference.gross_fusion_mw,
        useful_power_mw=reference.useful_power_mw,
        bremsstrahlung_loss_mw=reference.bremsstrahlung_loss_mw,
        transport_loss_mw=reference.transport_loss_mw,
        magnetic_energy_mj=reference.magnetic_energy_mj,
        magnet_mass_proxy_tonnes=reference.magnet_mass_proxy_tonnes,
        required_field_t=reference.required_field_t,
        separatrix_radius_m=reference.separatrix_radius_m,
        length_m=reference.length_m,
        temperature_kev=reference.temperature_kev,
        ion_density_m3=reference.ion_density_m3,
        confinement_s=reference.confinement_s,
        closes_engineering_net=plant_net_power_mw > 0.0,
        warnings=warnings,
    )


def run_engineering_net_budget(
    assumptions: Optional[EngineeringAssumptions] = None,
    limit: int = 12,
) -> List[EngineeringNetResult]:
    """Evaluate engineering net power for top LunarFire reference candidates."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    references = solve_reference_design(limit=limit)
    results = [
        evaluate_engineering_net(reference, assumptions=assumptions)
        for reference in references
    ]
    return sorted(results, key=lambda result: result.plant_net_power_mw, reverse=True)


def _radiator_area_m2(
    rejected_heat_mw: float,
    radiator_temperature_k: float,
    emissivity: float,
) -> float:
    heat_w = rejected_heat_mw * 1.0e6
    flux_w_m2 = emissivity * STEFAN_BOLTZMANN_W_M2_K4 * radiator_temperature_k**4
    return heat_w / flux_w_m2


def _warnings(reference: ReferenceDesignResult, plant_net_power_mw: float) -> str:
    warnings = [warning for warning in reference.warnings.split("; ") if warning]
    warnings.append("plant-net includes first-order engineering loads")
    if plant_net_power_mw <= 0.0:
        warnings.append("does not close plant-net power")
    if reference.bremsstrahlung_loss_mw > 10.0 * reference.screening_net_power_mw:
        warnings.append("bremsstrahlung dominates engineering design")
    return "; ".join(warnings)
