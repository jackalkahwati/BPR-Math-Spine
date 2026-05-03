"""Simplified plasma operating-point calculations for Helionis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from helionis.constants import KEV_TO_J
from helionis.reactions import D_D_AVERAGE, D_HE3, D_T, Reaction, get_reaction


@dataclass(frozen=True)
class OperatingPoint:
    """Minimal plasma state for an order-of-magnitude trade study."""

    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    plasma_volume_m3: float
    electron_density_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.temperature_kev <= 0:
            raise ValueError("temperature_kev must be positive")
        if self.ion_density_m3 <= 0:
            raise ValueError("ion_density_m3 must be positive")
        if self.confinement_s <= 0:
            raise ValueError("confinement_s must be positive")
        if self.plasma_volume_m3 <= 0:
            raise ValueError("plasma_volume_m3 must be positive")
        if self.electron_density_multiplier <= 0:
            raise ValueError("electron_density_multiplier must be positive")

    @property
    def triple_product(self) -> float:
        """n*T*tau in keV*s/m^3, a Lawson-style screening metric."""
        return self.ion_density_m3 * self.temperature_kev * self.confinement_s

    @property
    def electron_density_m3(self) -> float:
        """Approximate electron density for a quasi-neutral mixed-fuel plasma."""
        return self.ion_density_m3 * self.electron_density_multiplier

    @property
    def thermal_energy_density_j_m3(self) -> float:
        """Thermal plasma energy density using ion and electron populations."""
        total_particles = self.ion_density_m3 + self.electron_density_m3
        return 1.5 * total_particles * self.temperature_kev * KEV_TO_J


# Coarse Maxwellian reactivity anchors [m^3/s]. Values are deliberately rounded
# for screening studies; use Bosch-Hale or a kinetic solver before design claims.
REACTIVITY_ANCHORS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    D_HE3.key: (
        np.array([30.0, 50.0, 70.0, 100.0, 150.0, 200.0]),
        np.array([6.0e-24, 4.0e-23, 1.1e-22, 2.5e-22, 5.0e-22, 7.0e-22]),
    ),
    D_T.key: (
        np.array([5.0, 10.0, 15.0, 20.0, 30.0, 50.0]),
        np.array([1.5e-23, 1.1e-22, 2.4e-22, 4.2e-22, 8.7e-22, 1.8e-21]),
    ),
    D_D_AVERAGE.key: (
        np.array([20.0, 30.0, 50.0, 70.0, 100.0, 150.0]),
        np.array([1.0e-24, 3.0e-24, 1.0e-23, 2.0e-23, 4.0e-23, 8.0e-23]),
    ),
}


def estimate_reactivity(reaction_key: str, temperature_kev: float) -> float:
    """Estimate Maxwellian reactivity from rounded log-log anchor points."""
    if temperature_kev <= 0:
        raise ValueError("temperature_kev must be positive")
    if reaction_key not in REACTIVITY_ANCHORS:
        valid = ", ".join(sorted(REACTIVITY_ANCHORS))
        raise ValueError(f"Unknown reactivity key '{reaction_key}'. Valid keys: {valid}")

    temps, reactivities = REACTIVITY_ANCHORS[reaction_key]
    log_reactivity = np.interp(
        np.log(temperature_kev),
        np.log(temps),
        np.log(reactivities),
        left=np.log(reactivities[0]),
        right=np.log(reactivities[-1]),
    )
    return float(np.exp(log_reactivity))


def reactivity_temperature_range(reaction_key: str) -> Tuple[float, float]:
    """Return the modeled temperature range for a reaction reactivity table."""
    if reaction_key not in REACTIVITY_ANCHORS:
        valid = ", ".join(sorted(REACTIVITY_ANCHORS))
        raise ValueError(f"Unknown reactivity key '{reaction_key}'. Valid keys: {valid}")
    temps, _ = REACTIVITY_ANCHORS[reaction_key]
    return float(temps[0]), float(temps[-1])


def is_temperature_in_reactivity_range(
    reaction_key: str,
    temperature_kev: float,
) -> bool:
    """Return whether a temperature is inside the rounded anchor range."""
    low, high = reactivity_temperature_range(reaction_key)
    return low <= temperature_kev <= high


def reacting_pair_densities(
    reaction: Reaction,
    ion_density_m3: float,
    fuel_a_fraction: float = 0.5,
) -> Tuple[float, float]:
    """Split total ion density into reacting species densities."""
    if not 0.0 < fuel_a_fraction < 1.0:
        raise ValueError("fuel_a_fraction must be between 0 and 1")

    if reaction.same_species:
        return ion_density_m3, ion_density_m3

    n_a = ion_density_m3 * fuel_a_fraction
    n_b = ion_density_m3 * (1.0 - fuel_a_fraction)
    return n_a, n_b


def reaction_rate_density(
    reaction_key: str,
    operating_point: OperatingPoint,
    fuel_a_fraction: float = 0.5,
) -> float:
    """Return fusion reactions per m^3 per second for the operating point."""
    reaction = get_reaction(reaction_key)
    n_a, n_b = reacting_pair_densities(
        reaction,
        operating_point.ion_density_m3,
        fuel_a_fraction=fuel_a_fraction,
    )
    pair_factor = 0.5 if reaction.same_species else 1.0
    return (
        pair_factor
        * n_a
        * n_b
        * estimate_reactivity(reaction_key, operating_point.temperature_kev)
    )
