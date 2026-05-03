"""Zero-shot reactor geometry screening for Helionis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite, pi, sqrt
from typing import Dict, Iterable, List, Optional

from helionis.architecture import Scenario, evaluate_scenario
from helionis.constants import KEV_TO_J, MODEL_LABEL, MU_0
from helionis.reactions import D_HE3


class GeometryFamily(str, Enum):
    """Geometry families considered in the first Helionis downselect."""

    FRC = "frc"
    MIRROR = "mirror"
    SPHERICAL_TORUS = "spherical_torus"


@dataclass(frozen=True)
class GeometryCandidate:
    """Input geometry assumptions for a zero-shot reactor screen."""

    name: str
    family: GeometryFamily
    volume_m3: float
    major_radius_m: float
    minor_radius_m: float
    length_m: float
    beta_target: float
    shape_factor: float
    compactness_weight: float
    direct_conversion_access: float
    stability_confidence: float
    engineering_simplicity: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.family, GeometryFamily):
            raise ValueError("family must be a GeometryFamily")
        for field in (
            "volume_m3",
            "major_radius_m",
            "minor_radius_m",
            "length_m",
            "beta_target",
            "shape_factor",
        ):
            value = getattr(self, field)
            if not isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.volume_m3 <= 0:
            raise ValueError("volume_m3 must be positive")
        if self.major_radius_m <= 0 or self.minor_radius_m <= 0:
            raise ValueError("geometry radii must be positive")
        if self.length_m <= 0:
            raise ValueError("length_m must be positive")
        if not 0.0 < self.beta_target <= 1.0:
            raise ValueError("beta_target must be between 0 and 1")
        if self.shape_factor <= 0:
            raise ValueError("shape_factor must be positive")
        for field in (
            "compactness_weight",
            "direct_conversion_access",
            "stability_confidence",
            "engineering_simplicity",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")


@dataclass(frozen=True)
class GeometryScore:
    """Shared scoring output for comparing reactor geometry families."""

    scenario_name: str
    model_label: str
    family: str
    candidate_name: str
    volume_m3: float
    major_radius_m: float
    minor_radius_m: float
    length_m: float
    plasma_pressure_pa: float
    beta: float
    required_field_t: float
    magnetic_pressure_pa: float
    magnetic_energy_mj: float
    surface_area_m2: float
    neutron_wall_load_mw_m2: float
    magnet_mass_proxy_tonnes: float
    direct_conversion_access: float
    compactness_score: float
    stability_score: float
    engineering_score: float
    net_power_score: float
    shielding_score: float
    magnetic_score: float
    total_score: float
    rationale: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV/pandas-friendly row."""
        return asdict(self)


def plasma_pressure(scenario: Scenario) -> float:
    """Return total ion+electron pressure for a scenario in pascals."""
    electron_multiplier = _electron_density_multiplier(scenario)
    total_particles_m3 = scenario.ion_density_m3 * (1.0 + electron_multiplier)
    return total_particles_m3 * scenario.temperature_kev * KEV_TO_J


def magnetic_pressure(field_t: float) -> float:
    """Return magnetic pressure in pascals."""
    if field_t <= 0:
        raise ValueError("field_t must be positive")
    return field_t**2 / (2.0 * MU_0)


def magnetic_field_for_beta(pressure_pa: float, beta: float) -> float:
    """Return magnetic field needed to confine pressure at beta."""
    if pressure_pa <= 0:
        raise ValueError("pressure_pa must be positive")
    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")
    return sqrt(2.0 * MU_0 * pressure_pa / beta)


def build_geometry_candidates(scenario: Scenario) -> List[GeometryCandidate]:
    """Build comparable FRC, mirror, and spherical-torus candidates."""
    volume = scenario.plasma_volume_m3
    return [
        _frc_candidate(volume),
        _mirror_candidate(volume),
        _spherical_torus_candidate(volume),
    ]


def evaluate_geometry_candidate(
    candidate: GeometryCandidate,
    scenario: Scenario,
) -> GeometryScore:
    """Score one geometry candidate against one Helionis scenario."""
    pressure_pa = plasma_pressure(scenario)
    field_t = magnetic_field_for_beta(pressure_pa, candidate.beta_target)
    mag_pressure = magnetic_pressure(field_t)
    trade = evaluate_scenario(scenario)

    surface_area = _surface_area(candidate)
    neutron_wall_load = (
        trade.neutron_power_mw / surface_area if surface_area > 0.0 else float("inf")
    )
    magnetic_energy_mj = mag_pressure * candidate.volume_m3 / 1.0e6
    magnet_mass_proxy = _magnet_mass_proxy(candidate, field_t, magnetic_energy_mj)

    compactness = _compactness_score(candidate)
    stability = candidate.stability_confidence
    engineering = candidate.engineering_simplicity
    shielding = _bounded_inverse(neutron_wall_load, scale=1.0)
    net_power = _bounded_score(trade.gain_proxy, target=1.0)
    magnetic = _bounded_inverse(magnet_mass_proxy, scale=500.0)

    total = (
        0.22 * candidate.direct_conversion_access
        + 0.18 * compactness
        + 0.13 * stability
        + 0.12 * engineering
        + 0.13 * shielding
        + 0.07 * net_power
        + 0.15 * magnetic
    )

    return GeometryScore(
        scenario_name=scenario.name,
        model_label=MODEL_LABEL,
        family=candidate.family.value,
        candidate_name=candidate.name,
        volume_m3=candidate.volume_m3,
        major_radius_m=candidate.major_radius_m,
        minor_radius_m=candidate.minor_radius_m,
        length_m=candidate.length_m,
        plasma_pressure_pa=pressure_pa,
        beta=candidate.beta_target,
        required_field_t=field_t,
        magnetic_pressure_pa=mag_pressure,
        magnetic_energy_mj=magnetic_energy_mj,
        surface_area_m2=surface_area,
        neutron_wall_load_mw_m2=neutron_wall_load,
        magnet_mass_proxy_tonnes=magnet_mass_proxy,
        direct_conversion_access=candidate.direct_conversion_access,
        compactness_score=compactness,
        stability_score=stability,
        engineering_score=engineering,
        net_power_score=net_power,
        shielding_score=shielding,
        magnetic_score=magnetic,
        total_score=total,
        rationale=_rationale(candidate),
    )


def rank_geometry_candidates(
    scenario: Scenario,
    candidates: Optional[Iterable[GeometryCandidate]] = None,
) -> List[GeometryScore]:
    """Return geometry candidates ranked best-first for a scenario."""
    candidate_list = (
        list(candidates) if candidates is not None else build_geometry_candidates(scenario)
    )
    scores = [
        evaluate_geometry_candidate(candidate, scenario)
        for candidate in candidate_list
    ]
    return sorted(scores, key=lambda score: score.total_score, reverse=True)


def run_geometry_downselect(scenarios: Iterable[Scenario]) -> List[GeometryScore]:
    """Rank all geometry families across supplied scenarios."""
    rows: List[GeometryScore] = []
    for scenario in scenarios:
        if scenario.reaction_key == D_HE3.key:
            rows.extend(rank_geometry_candidates(scenario))
    return rows


def _frc_candidate(volume_m3: float) -> GeometryCandidate:
    aspect = 6.0
    shape_factor = 0.85
    radius = (volume_m3 / (pi * aspect * shape_factor)) ** (1.0 / 3.0)
    length = aspect * radius
    return GeometryCandidate(
        name="frc_linear_plasmoid",
        family=GeometryFamily.FRC,
        volume_m3=volume_m3,
        major_radius_m=length / 2.0,
        minor_radius_m=radius,
        length_m=length,
        beta_target=0.85,
        shape_factor=shape_factor,
        compactness_weight=0.85,
        direct_conversion_access=0.95,
        stability_confidence=0.55,
        engineering_simplicity=0.78,
        notes="High-beta compact plasmoid with linear end access.",
    )


def _mirror_candidate(volume_m3: float) -> GeometryCandidate:
    aspect = 10.0
    radius = (volume_m3 / (pi * aspect)) ** (1.0 / 3.0)
    length = aspect * radius
    return GeometryCandidate(
        name="linear_mirror_nozzle",
        family=GeometryFamily.MIRROR,
        volume_m3=volume_m3,
        major_radius_m=length / 2.0,
        minor_radius_m=radius,
        length_m=length,
        beta_target=0.45,
        shape_factor=1.0,
        compactness_weight=0.62,
        direct_conversion_access=0.88,
        stability_confidence=0.45,
        engineering_simplicity=0.72,
        notes="Linear extraction-friendly geometry with end-loss burden.",
    )


def _spherical_torus_candidate(volume_m3: float) -> GeometryCandidate:
    aspect = 1.7
    elongation = 2.0
    minor = (volume_m3 / (2.0 * pi**2 * aspect * elongation)) ** (1.0 / 3.0)
    major = aspect * minor
    outer_diameter = 2.0 * (major + minor)
    return GeometryCandidate(
        name="spherical_torus_reference",
        family=GeometryFamily.SPHERICAL_TORUS,
        volume_m3=volume_m3,
        major_radius_m=major,
        minor_radius_m=minor,
        length_m=outer_diameter,
        beta_target=0.28,
        shape_factor=elongation,
        compactness_weight=0.55,
        direct_conversion_access=0.32,
        stability_confidence=0.70,
        engineering_simplicity=0.42,
        notes="Mature toroidal baseline, less aligned with direct conversion.",
    )


def _surface_area(candidate: GeometryCandidate) -> float:
    if candidate.family == GeometryFamily.SPHERICAL_TORUS:
        minor = candidate.minor_radius_m
        elongated = candidate.shape_factor * minor
        perimeter = pi * (
            3.0 * (minor + elongated)
            - sqrt((3.0 * minor + elongated) * (minor + 3.0 * elongated))
        )
        return 2.0 * pi * candidate.major_radius_m * perimeter
    radius = candidate.minor_radius_m
    return 2.0 * pi * radius * candidate.length_m + 2.0 * pi * radius**2


def _magnet_mass_proxy(
    candidate: GeometryCandidate,
    field_t: float,
    magnetic_energy_mj: float,
) -> float:
    geometry_multiplier = {
        GeometryFamily.FRC: 0.75,
        GeometryFamily.MIRROR: 0.90,
        GeometryFamily.SPHERICAL_TORUS: 1.45,
    }[candidate.family]
    return geometry_multiplier * (0.35 * magnetic_energy_mj + 2.0 * field_t)


def _compactness_score(candidate: GeometryCandidate) -> float:
    if candidate.family == GeometryFamily.SPHERICAL_TORUS:
        envelope = candidate.length_m
    else:
        envelope = candidate.length_m + 2.0 * candidate.minor_radius_m
    return candidate.compactness_weight * _bounded_inverse(envelope, scale=20.0)


def _bounded_inverse(value: float, scale: float) -> float:
    if value < 0:
        raise ValueError("value cannot be negative")
    return scale / (scale + value)


def _bounded_score(value: float, target: float) -> float:
    if value <= 0:
        return 0.0
    return min(value / target, 1.0)


def _electron_density_multiplier(scenario: Scenario) -> float:
    if scenario.electron_density_multiplier is not None:
        return scenario.electron_density_multiplier
    if scenario.reaction_key == D_HE3.key:
        return scenario.fuel_a_fraction + 2.0 * (1.0 - scenario.fuel_a_fraction)
    return 1.0


def _rationale(candidate: GeometryCandidate) -> str:
    if candidate.family == GeometryFamily.FRC:
        return "Best fit for high-beta compact D-He3 with direct end access."
    if candidate.family == GeometryFamily.MIRROR:
        return "Strong direct-conversion access, penalized for end-loss confinement risk."
    return "Mature equilibrium baseline, penalized for toroidal mass and poor charged-product access."
