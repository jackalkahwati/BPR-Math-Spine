"""Loss, conversion, and shielding proxies for Helionis trade studies."""

from __future__ import annotations

from dataclasses import dataclass

from helionis.constants import (
    BREMSSTRAHLUNG_COEFF,
    DEFAULT_BASE_SHIELDING_TONNES,
    DEFAULT_SHIELDING_TONNES_PER_NEUTRON_MW,
)
from helionis.plasma import OperatingPoint


@dataclass(frozen=True)
class PowerBalance:
    """Power-density accounting for a reactor architecture screening point."""

    fusion_power_density_w_m3: float
    charged_power_density_w_m3: float
    neutron_power_density_w_m3: float
    bremsstrahlung_loss_density_w_m3: float
    transport_loss_density_w_m3: float
    useful_power_density_w_m3: float

    @property
    def total_loss_density_w_m3(self) -> float:
        """Total modeled plasma loss density."""
        return self.bremsstrahlung_loss_density_w_m3 + self.transport_loss_density_w_m3

    @property
    def net_power_density_w_m3(self) -> float:
        """Useful electrical output minus modeled plasma losses."""
        return self.useful_power_density_w_m3 - self.total_loss_density_w_m3

    @property
    def gain_proxy(self) -> float:
        """Useful/loss ratio; not Q_plasma and not a net-energy claim."""
        if self.total_loss_density_w_m3 <= 0:
            return float("inf")
        return self.useful_power_density_w_m3 / self.total_loss_density_w_m3


def bremsstrahlung_loss_density(
    operating_point: OperatingPoint,
    z_eff: float = 1.5,
) -> float:
    """Approximate bremsstrahlung loss density in W/m^3."""
    if z_eff <= 0:
        raise ValueError("z_eff must be positive")

    temperature_ev = operating_point.temperature_kev * 1.0e3
    return (
        BREMSSTRAHLUNG_COEFF
        * z_eff
        * operating_point.electron_density_m3
        * operating_point.ion_density_m3
        * temperature_ev**0.5
    )


def transport_loss_density(
    operating_point: OperatingPoint,
    multiplier: float = 1.0,
) -> float:
    """Approximate non-radiative thermal loss density in W/m^3."""
    if multiplier < 0:
        raise ValueError("transport loss multiplier cannot be negative")
    return multiplier * (
        operating_point.thermal_energy_density_j_m3 / operating_point.confinement_s
    )


def useful_power_density(
    charged_power_density_w_m3: float,
    neutron_power_density_w_m3: float,
    direct_conversion_efficiency: float,
    thermal_conversion_efficiency: float,
) -> float:
    """Convert charged and neutron/thermal channels into useful power density."""
    for name, value in {
        "direct_conversion_efficiency": direct_conversion_efficiency,
        "thermal_conversion_efficiency": thermal_conversion_efficiency,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    direct_charged = charged_power_density_w_m3 * direct_conversion_efficiency
    thermalized_power = (
        charged_power_density_w_m3 * (1.0 - direct_conversion_efficiency)
        + neutron_power_density_w_m3
    )
    return direct_charged + thermalized_power * thermal_conversion_efficiency


def shielding_mass_proxy_tonnes(
    neutron_power_mw: float,
    base_tonnes: float = DEFAULT_BASE_SHIELDING_TONNES,
    tonnes_per_neutron_mw: float = DEFAULT_SHIELDING_TONNES_PER_NEUTRON_MW,
) -> float:
    """Return a comparative shielding mass proxy for neutron-producing designs."""
    if neutron_power_mw < 0:
        raise ValueError("neutron_power_mw cannot be negative")
    if base_tonnes < 0 or tonnes_per_neutron_mw < 0:
        raise ValueError("shielding coefficients cannot be negative")
    return base_tonnes + neutron_power_mw * tonnes_per_neutron_mw
