"""LunarFire v0.9 BPR-coupled mirror/nozzle screening."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from numbers import Real
from typing import Dict, List, Optional, Sequence

from helionis.architecture_comparison_v07 import (
    ArchitectureV07Assumptions,
    _candidate,
    mirror_leakage_transport_multiplier,
)
from helionis.bpr_bridge import BPRBridgeInputs, bpr_bridge_factors
from helionis.geometry import GeometryFamily
from helionis.mirror_nozzle_v06 import charged_product_collector_efficiency


DEFAULT_V09_TEMPERATURES_KEV = (200.0,)
DEFAULT_V09_DENSITIES_M3 = (2.5e20, 3.5e20)
DEFAULT_V09_CONFINEMENTS_S = (16.0, 20.0)
DEFAULT_V09_MIRROR_RATIOS = (4.0, 5.0)
DEFAULT_V09_MIRROR_ASPECT_RATIOS = (8.0, 10.0, 12.0)
DEFAULT_V09_ALPHA_COLLECTOR_KV = (1800.0,)
DEFAULT_V09_PROTON_COLLECTOR_KV = (15000.0,)


@dataclass(frozen=True)
class BPRCoupledV09Result:
    """One BPR-coupled mirror/nozzle candidate."""

    target_screening_net_mw: float
    plant_net_power_mw: float
    closes_engineering_net: bool
    uses_bpr_math: bool
    base_transport_multiplier: float
    adjusted_transport_multiplier: float
    base_direct_conversion_efficiency: float
    adjusted_direct_conversion_efficiency: float
    bpr_impedance_match: float
    bpr_resonance_alignment: float
    bpr_transport_multiplier: float
    bpr_direct_conversion_multiplier: float
    gross_fusion_mw: float
    engineering_load_mw: float
    rejected_heat_mw: float
    radiator_area_m2: float
    mirror_ratio: float
    mirror_aspect_ratio: float
    plug_field_t: float
    separatrix_radius_m: float
    plasma_length_m: float
    plasma_volume_m3: float
    collector_area_m2: float
    temperature_kev: float
    ion_density_m3: float
    confinement_s: float
    bpr_source_modules: str
    warnings: str

    def to_row(self) -> Dict[str, object]:
        """Return a CSV-friendly row."""
        return asdict(self)


def run_bpr_coupled_v09_sweep(
    target_screening_net_mw: float = 50.0,
    assumptions: Optional[ArchitectureV07Assumptions] = None,
    temperatures_kev: Sequence[float] = DEFAULT_V09_TEMPERATURES_KEV,
    densities_m3: Sequence[float] = DEFAULT_V09_DENSITIES_M3,
    confinements_s: Sequence[float] = DEFAULT_V09_CONFINEMENTS_S,
    mirror_ratios: Sequence[float] = DEFAULT_V09_MIRROR_RATIOS,
    mirror_aspect_ratios: Sequence[float] = DEFAULT_V09_MIRROR_ASPECT_RATIOS,
    alpha_collector_voltages_kv: Sequence[float] = DEFAULT_V09_ALPHA_COLLECTOR_KV,
    proton_collector_voltages_kv: Sequence[float] = DEFAULT_V09_PROTON_COLLECTOR_KV,
    limit: Optional[int] = 12,
) -> List[BPRCoupledV09Result]:
    """Run a BPR-coupled mirror/nozzle sweep."""
    _require_positive_finite_number("target_screening_net_mw", target_screening_net_mw)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
    assumptions = assumptions or ArchitectureV07Assumptions()
    _validate_values("temperatures_kev", temperatures_kev)
    _validate_values("densities_m3", densities_m3)
    _validate_values("confinements_s", confinements_s)
    _validate_mirror_ratios(mirror_ratios)
    _validate_values("mirror_aspect_ratios", mirror_aspect_ratios)
    _validate_values("alpha_collector_voltages_kv", alpha_collector_voltages_kv)
    _validate_values("proton_collector_voltages_kv", proton_collector_voltages_kv)

    rows: List[BPRCoupledV09Result] = []
    for mirror_aspect_ratio in mirror_aspect_ratios:
        local_assumptions = replace(
            assumptions,
            mirror_aspect_ratio=mirror_aspect_ratio,
        )
        for temperature_kev in temperatures_kev:
            for density_m3 in densities_m3:
                for confinement_s in confinements_s:
                    for mirror_ratio in mirror_ratios:
                        base_transport = mirror_leakage_transport_multiplier(
                            mirror_ratio=mirror_ratio,
                            confinement_s=confinement_s,
                            pitch_angle_scattering_s=local_assumptions.pitch_angle_scattering_s,
                            assumptions=local_assumptions,
                        )
                        for alpha_voltage in alpha_collector_voltages_kv:
                            for proton_voltage in proton_collector_voltages_kv:
                                base_direct = min(
                                    charged_product_collector_efficiency(
                                        alpha_voltage,
                                        proton_voltage,
                                    )
                                    + _nozzle_expansion_bonus(
                                        mirror_ratio, local_assumptions
                                    ),
                                    local_assumptions.max_direct_conversion_efficiency,
                                )
                                bridge = bpr_bridge_factors(
                                    BPRBridgeInputs(
                                        mirror_ratio=mirror_ratio,
                                        radius_m=1.0,
                                        length_m=mirror_aspect_ratio,
                                        base_transport_multiplier=base_transport,
                                        base_direct_conversion_efficiency=base_direct,
                                    )
                                )
                                adjusted_transport = (
                                    base_transport * bridge.transport_multiplier
                                )
                                adjusted_direct = min(
                                    local_assumptions.max_direct_conversion_efficiency,
                                    base_direct * bridge.direct_conversion_multiplier,
                                )
                                adjusted_row = _candidate(
                                    family=GeometryFamily.MIRROR,
                                    target_screening_net_mw=target_screening_net_mw,
                                    assumptions=local_assumptions,
                                    temperature_kev=temperature_kev,
                                    density_m3=density_m3,
                                    confinement_s=confinement_s,
                                    mirror_ratio=mirror_ratio,
                                    alpha_collector_voltage_kv=alpha_voltage,
                                    proton_collector_voltage_kv=proton_voltage,
                                    direct_efficiency=adjusted_direct,
                                    transport_loss_multiplier=adjusted_transport,
                                )
                                if adjusted_row is not None:
                                    rows.append(
                                        _result(
                                            target_screening_net_mw,
                                            adjusted_row,
                                            base_transport,
                                            base_direct,
                                            adjusted_transport,
                                            adjusted_direct,
                                            bridge,
                                            mirror_aspect_ratio,
                                        )
                                    )

    ranked = sorted(rows, key=lambda row: row.plant_net_power_mw, reverse=True)
    return ranked[:limit] if limit is not None else ranked


def _result(
    target_screening_net_mw: float,
    row,
    base_transport: float,
    base_direct: float,
    adjusted_transport: float,
    adjusted_direct: float,
    bridge,
    mirror_aspect_ratio: float,
) -> BPRCoupledV09Result:
    return BPRCoupledV09Result(
        target_screening_net_mw=target_screening_net_mw,
        plant_net_power_mw=row.plant_net_power_mw,
        closes_engineering_net=row.closes_engineering_net,
        uses_bpr_math=True,
        base_transport_multiplier=base_transport,
        adjusted_transport_multiplier=adjusted_transport,
        base_direct_conversion_efficiency=base_direct,
        adjusted_direct_conversion_efficiency=adjusted_direct,
        bpr_impedance_match=bridge.impedance_match,
        bpr_resonance_alignment=bridge.resonance_alignment,
        bpr_transport_multiplier=bridge.transport_multiplier,
        bpr_direct_conversion_multiplier=bridge.direct_conversion_multiplier,
        gross_fusion_mw=row.gross_fusion_mw,
        engineering_load_mw=row.engineering_load_mw,
        rejected_heat_mw=row.rejected_heat_mw,
        radiator_area_m2=row.radiator_area_m2,
        mirror_ratio=row.mirror_ratio,
        mirror_aspect_ratio=mirror_aspect_ratio,
        plug_field_t=row.plug_field_t,
        separatrix_radius_m=row.separatrix_radius_m,
        plasma_length_m=row.length_m,
        plasma_volume_m3=row.plasma_volume_m3,
        collector_area_m2=row.collector_area_m2,
        temperature_kev=row.temperature_kev,
        ion_density_m3=row.ion_density_m3,
        confinement_s=row.confinement_s,
        bpr_source_modules=bridge.source_modules,
        warnings=f"{row.warnings}; v0.9 uses bounded BPR bridge factors",
    )


def _nozzle_expansion_bonus(
    mirror_ratio: float,
    assumptions: ArchitectureV07Assumptions,
) -> float:
    return assumptions.nozzle_expansion_bonus * min((mirror_ratio - 1.0) / 4.0, 1.0)


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
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
