"""Scenario architecture layer for Helionis D-He3 trade studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from helionis.constants import (
    DEFAULT_BASE_SHIELDING_TONNES,
    DEFAULT_SHIELDING_TONNES_PER_NEUTRON_MW,
    MODEL_LABEL,
)
from helionis.losses import (
    PowerBalance,
    bremsstrahlung_loss_density,
    shielding_mass_proxy_tonnes,
    transport_loss_density,
    useful_power_density,
)
from helionis.plasma import (
    OperatingPoint,
    estimate_reactivity,
    is_temperature_in_reactivity_range,
    reaction_rate_density,
    reactivity_temperature_range,
)
from helionis.reactions import D_D_AVERAGE, D_HE3, D_T, get_reaction


@dataclass(frozen=True)
class Scenario:
    """Input assumptions for a compact fusion architecture screening case."""

    name: str
    reaction_key: str
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    plasma_volume_m3: float
    direct_conversion_efficiency: float
    thermal_conversion_efficiency: float
    target_power_mw: float = 25.0
    fuel_a_fraction: float = 0.5
    electron_density_multiplier: Optional[float] = None
    z_eff: float = 1.5
    transport_loss_multiplier: float = 1.0
    dd_side_reaction_fraction: float = 0.0
    base_shielding_tonnes: float = DEFAULT_BASE_SHIELDING_TONNES
    shielding_tonnes_per_neutron_mw: float = DEFAULT_SHIELDING_TONNES_PER_NEUTRON_MW
    notes: str = ""

    def __post_init__(self) -> None:
        if self.temperature_kev <= 0:
            raise ValueError("temperature_kev must be positive")
        if self.ion_density_m3 <= 0:
            raise ValueError("ion_density_m3 must be positive")
        if self.confinement_s <= 0:
            raise ValueError("confinement_s must be positive")
        if self.plasma_volume_m3 <= 0:
            raise ValueError("plasma_volume_m3 must be positive")
        if self.target_power_mw <= 0:
            raise ValueError("target_power_mw must be positive")
        if not 0.0 < self.fuel_a_fraction < 1.0:
            raise ValueError("fuel_a_fraction must be between 0 and 1")
        if self.electron_density_multiplier is not None:
            if self.electron_density_multiplier <= 0:
                raise ValueError("electron_density_multiplier must be positive")
        if not 0.0 <= self.direct_conversion_efficiency <= 1.0:
            raise ValueError("direct_conversion_efficiency must be between 0 and 1")
        if not 0.0 <= self.thermal_conversion_efficiency <= 1.0:
            raise ValueError("thermal_conversion_efficiency must be between 0 and 1")
        if self.transport_loss_multiplier < 0:
            raise ValueError("transport_loss_multiplier cannot be negative")
        if not 0.0 <= self.dd_side_reaction_fraction <= 1.0:
            raise ValueError("dd_side_reaction_fraction must be between 0 and 1")
        if self.base_shielding_tonnes < 0:
            raise ValueError("base_shielding_tonnes cannot be negative")
        if self.shielding_tonnes_per_neutron_mw < 0:
            raise ValueError("shielding_tonnes_per_neutron_mw cannot be negative")


@dataclass(frozen=True)
class TradeStudyResult:
    """Computed output row for a Helionis architecture scenario."""

    name: str
    model_label: str
    reaction: str
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    plasma_volume_m3: float
    triple_product_kev_s_m3: float
    reactivity_m3_s: float
    fusion_power_mw: float
    charged_power_mw: float
    neutron_power_mw: float
    neutron_fraction_of_fusion_power: float
    bremsstrahlung_loss_mw: float
    transport_loss_mw: float
    useful_power_mw: float
    net_power_mw: float
    gain_proxy: float
    required_volume_for_target_m3: float
    shielding_mass_proxy_tonnes: float
    warnings: str
    notes: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


DEFAULT_SCENARIOS = [
    Scenario(
        name="lunar_infrastructure_dhe3",
        reaction_key=D_HE3.key,
        temperature_kev=100.0,
        ion_density_m3=2.5e20,
        confinement_s=5.0,
        plasma_volume_m3=80.0,
        direct_conversion_efficiency=0.65,
        thermal_conversion_efficiency=0.38,
        target_power_mw=25.0,
        dd_side_reaction_fraction=0.05,
        notes="Compact lunar base module using imported or locally supplied He3.",
    ),
    Scenario(
        name="orbital_data_center_dhe3",
        reaction_key=D_HE3.key,
        temperature_kev=120.0,
        ion_density_m3=3.0e20,
        confinement_s=6.0,
        plasma_volume_m3=180.0,
        direct_conversion_efficiency=0.70,
        thermal_conversion_efficiency=0.38,
        target_power_mw=100.0,
        dd_side_reaction_fraction=0.04,
        notes="High-density power architecture for orbital compute loads.",
    ),
    Scenario(
        name="compact_space_reactor_dhe3",
        reaction_key=D_HE3.key,
        temperature_kev=90.0,
        ion_density_m3=2.0e20,
        confinement_s=3.0,
        plasma_volume_m3=40.0,
        direct_conversion_efficiency=0.60,
        thermal_conversion_efficiency=0.35,
        target_power_mw=10.0,
        dd_side_reaction_fraction=0.07,
        notes="Mobile space-power case where shielding mass matters most.",
    ),
    Scenario(
        name="terrestrial_demonstrator_dhe3",
        reaction_key=D_HE3.key,
        temperature_kev=150.0,
        ion_density_m3=2.5e20,
        confinement_s=8.0,
        plasma_volume_m3=120.0,
        direct_conversion_efficiency=0.72,
        thermal_conversion_efficiency=0.40,
        target_power_mw=50.0,
        dd_side_reaction_fraction=0.03,
        notes="Aggressive subsystem demonstrator assumption set.",
    ),
    Scenario(
        name="dt_reference_power_block",
        reaction_key=D_T.key,
        temperature_kev=15.0,
        ion_density_m3=1.5e20,
        confinement_s=1.0,
        plasma_volume_m3=80.0,
        direct_conversion_efficiency=0.0,
        thermal_conversion_efficiency=0.38,
        target_power_mw=25.0,
        z_eff=1.0,
        notes="D-T reference block for neutron and shielding comparison.",
    ),
]


def evaluate_scenario(scenario: Scenario) -> TradeStudyResult:
    """Evaluate a Helionis scenario using transparent screening assumptions."""
    reaction = get_reaction(scenario.reaction_key)
    operating_point = OperatingPoint(
        temperature_kev=scenario.temperature_kev,
        ion_density_m3=scenario.ion_density_m3,
        confinement_s=scenario.confinement_s,
        plasma_volume_m3=scenario.plasma_volume_m3,
        electron_density_multiplier=_electron_density_multiplier(
            scenario.reaction_key,
            scenario.fuel_a_fraction,
            scenario.electron_density_multiplier,
        ),
    )

    primary_rate = reaction_rate_density(
        reaction.key,
        operating_point,
        fuel_a_fraction=scenario.fuel_a_fraction,
    )
    side_rate = 0.0
    if reaction.key == D_HE3.key and scenario.dd_side_reaction_fraction > 0.0:
        side_rate = primary_rate * scenario.dd_side_reaction_fraction

    fusion_density = primary_rate * reaction.q_j + side_rate * D_D_AVERAGE.q_j
    charged_density = (
        primary_rate * reaction.charged_j + side_rate * D_D_AVERAGE.charged_j
    )
    neutron_density = (
        primary_rate * reaction.neutron_j + side_rate * D_D_AVERAGE.neutron_j
    )

    brem_density = bremsstrahlung_loss_density(
        operating_point,
        z_eff=scenario.z_eff,
    )
    transport_density = transport_loss_density(
        operating_point,
        multiplier=scenario.transport_loss_multiplier,
    )
    useful_density = useful_power_density(
        charged_power_density_w_m3=charged_density,
        neutron_power_density_w_m3=neutron_density,
        direct_conversion_efficiency=scenario.direct_conversion_efficiency,
        thermal_conversion_efficiency=scenario.thermal_conversion_efficiency,
    )

    balance = PowerBalance(
        fusion_power_density_w_m3=fusion_density,
        charged_power_density_w_m3=charged_density,
        neutron_power_density_w_m3=neutron_density,
        bremsstrahlung_loss_density_w_m3=brem_density,
        transport_loss_density_w_m3=transport_density,
        useful_power_density_w_m3=useful_density,
    )

    volume = operating_point.plasma_volume_m3
    fusion_mw = fusion_density * volume / 1.0e6
    charged_mw = charged_density * volume / 1.0e6
    neutron_mw = neutron_density * volume / 1.0e6
    net_density = balance.net_power_density_w_m3
    required_volume = (
        scenario.target_power_mw * 1.0e6 / net_density
        if net_density > 0.0
        else float("inf")
    )

    warnings = _scenario_warnings(
        scenario=scenario,
        neutron_fraction=(
            neutron_density / fusion_density if fusion_density > 0.0 else 0.0
        ),
        net_power_mw=net_density * volume / 1.0e6,
    )

    return TradeStudyResult(
        name=scenario.name,
        model_label=MODEL_LABEL,
        reaction=reaction.name,
        temperature_kev=scenario.temperature_kev,
        ion_density_m3=scenario.ion_density_m3,
        confinement_s=scenario.confinement_s,
        plasma_volume_m3=volume,
        triple_product_kev_s_m3=operating_point.triple_product,
        reactivity_m3_s=estimate_reactivity(reaction.key, scenario.temperature_kev),
        fusion_power_mw=fusion_mw,
        charged_power_mw=charged_mw,
        neutron_power_mw=neutron_mw,
        neutron_fraction_of_fusion_power=(
            neutron_density / fusion_density if fusion_density > 0.0 else 0.0
        ),
        bremsstrahlung_loss_mw=brem_density * volume / 1.0e6,
        transport_loss_mw=transport_density * volume / 1.0e6,
        useful_power_mw=useful_density * volume / 1.0e6,
        net_power_mw=net_density * volume / 1.0e6,
        gain_proxy=balance.gain_proxy,
        required_volume_for_target_m3=required_volume,
        shielding_mass_proxy_tonnes=shielding_mass_proxy_tonnes(
            neutron_mw,
            base_tonnes=scenario.base_shielding_tonnes,
            tonnes_per_neutron_mw=scenario.shielding_tonnes_per_neutron_mw,
        ),
        warnings="; ".join(warnings),
        notes=scenario.notes,
    )


def run_trade_study(
    scenarios: Optional[Iterable[Scenario]] = None,
) -> List[TradeStudyResult]:
    """Evaluate all supplied scenarios, or the default Helionis MVP set."""
    return [evaluate_scenario(scenario) for scenario in scenarios or DEFAULT_SCENARIOS]


def _scenario_warnings(
    scenario: Scenario,
    neutron_fraction: float,
    net_power_mw: float,
) -> List[str]:
    warnings: List[str] = []
    if scenario.reaction_key == D_HE3.key and scenario.temperature_kev < 80.0:
        warnings.append("D-He3 temperature is below the intended screening window")
    if not is_temperature_in_reactivity_range(
        scenario.reaction_key,
        scenario.temperature_kev,
    ):
        low, high = reactivity_temperature_range(scenario.reaction_key)
        warnings.append(f"reactivity estimate clamped outside {low:.0f}-{high:.0f} keV")
    if scenario.reaction_key == D_HE3.key and scenario.dd_side_reaction_fraction > 0:
        warnings.append("D-D side reactions included as neutron source")
    if neutron_fraction > 0.5:
        warnings.append("high neutron power fraction")
    if net_power_mw <= 0:
        warnings.append("net power proxy is negative under current loss assumptions")
    return warnings


def _electron_density_multiplier(
    reaction_key: str,
    fuel_a_fraction: float,
    override: Optional[float],
) -> float:
    if override is not None:
        return override
    if reaction_key == D_HE3.key:
        return fuel_a_fraction * 1.0 + (1.0 - fuel_a_fraction) * 2.0
    return 1.0
