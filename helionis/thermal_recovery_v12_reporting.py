"""Reporting helpers for LunarFire v1.2 thermal packaging recovery."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.thermal_recovery_v12 import ThermalRecoveryV12Result


def write_thermal_recovery_v12_csv(
    results: Iterable[ThermalRecoveryV12Result],
    path: Path,
) -> None:
    """Write thermal recovery rows to CSV."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No thermal recovery rows to write")
    rows = [result.to_row() for result in result_list]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_thermal_recovery_v12_markdown(
    results: Iterable[ThermalRecoveryV12Result],
    path: Path,
    summary_results: Iterable[ThermalRecoveryV12Result] | None = None,
) -> None:
    """Write a thermal recovery summary."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No thermal recovery rows to summarize")
    summary_list = (
        _ranked(summary_results) if summary_results is not None else result_list
    )
    if not summary_list:
        raise ValueError("No thermal recovery summary rows to summarize")
    minimum_ready = next((row for row in summary_list if row.cad_ready), None)
    best_score_ready = max(
        (row for row in summary_list if row.cad_ready),
        key=lambda row: row.cad_readiness_score,
        default=None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v1.2 Thermal Packaging Recovery",
        "",
        _summary(minimum_ready, best_score_ready),
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- direct conversion recovery improves plant-net power and reduces rejected heat.",
        "- hotter radiators and topology packing improve packaging, but do not create net power.",
        "- CAD-ready means plant-net closes and radiator span/area fit the v1.2 constraints.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_thermal_recovery_v12(
    results: Iterable[ThermalRecoveryV12Result],
    path: Path,
) -> None:
    """Plot thermal recovery span versus plant-net power."""
    import matplotlib.pyplot as plt

    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No thermal recovery rows to plot")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#0f766e" if row.cad_ready else "#b45309" for row in result_list]
    ax.scatter(
        [row.adjusted_wing_span_each_m for row in result_list],
        [row.plant_net_power_mw for row in result_list],
        c=colors,
        alpha=0.8,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(500.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Adjusted radiator wing span per side (m)")
    ax.set_ylabel("Plant net power (MW)")
    ax.set_title("LunarFire v1.2 Thermal Packaging Recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ranked(
    results: Iterable[ThermalRecoveryV12Result],
) -> List[ThermalRecoveryV12Result]:
    return sorted(
        list(results),
        key=lambda row: (
            not row.cad_ready,
            row.recovery_aggressiveness,
            -row.cad_readiness_score,
        ),
    )


def _summary(
    minimum_ready: ThermalRecoveryV12Result | None,
    best_score_ready: ThermalRecoveryV12Result | None,
) -> str:
    if minimum_ready is None:
        return "CAD-ready thermal recovery recipe: `none found`."
    assert best_score_ready is not None
    return "\n".join(
        [
            "Minimum-assumption CAD-ready thermal recovery recipe:",
            "",
            f"- Plant-net power: `{minimum_ready.plant_net_power_mw:.1f} MW`",
            f"- Direct heat recovery: `{minimum_ready.direct_heat_recovery_fraction:.2f}`",
            f"- Radiator temperature: `{minimum_ready.radiator_temperature_k:.0f} K`",
            f"- Topology packing factor: `{minimum_ready.topology_packing_factor:.1f}`",
            f"- Adjusted radiator area: `{minimum_ready.adjusted_radiator_area_m2:.0f} m2`",
            f"- Adjusted wing span per side: `{minimum_ready.adjusted_wing_span_each_m:.0f} m`",
            "",
            f"Highest-score CAD-ready row: `{best_score_ready.cad_readiness_score:.3f}` "
            f"at `{best_score_ready.plant_net_power_mw:.1f} MW` plant-net.",
        ]
    )


def _markdown_table(results: List[ThermalRecoveryV12Result]) -> str:
    headers = [
        "Ready",
        "Plant MW",
        "Agg",
        "Recover frac",
        "Recovered MW",
        "Rad K",
        "Pack",
        "Heat MW",
        "Area m2",
        "Span m",
        "Blockers",
        "Source ID",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                str(result.cad_ready),
                f"{result.plant_net_power_mw:.1f}",
                f"{result.recovery_aggressiveness:.1f}",
                f"{result.direct_heat_recovery_fraction:.2f}",
                f"{result.recovered_electric_power_mw:.1f}",
                f"{result.radiator_temperature_k:.0f}",
                f"{result.topology_packing_factor:.1f}",
                f"{result.adjusted_rejected_heat_mw:.0f}",
                f"{result.adjusted_radiator_area_m2:.0f}",
                f"{result.adjusted_wing_span_each_m:.0f}",
                result.blockers,
                result.source_design_id,
            ]
        )
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)
