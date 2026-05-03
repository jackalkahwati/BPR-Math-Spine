"""Reporting helpers for LunarFire v1.3 collector/nozzle thermal architecture."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.collector_nozzle_thermal_v13 import CollectorNozzleThermalV13Result


def write_collector_nozzle_thermal_v13_csv(
    results: Iterable[CollectorNozzleThermalV13Result],
    path: Path,
) -> None:
    """Write collector/nozzle thermal rows to CSV."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No collector/nozzle thermal rows to write")
    rows = [result.to_row() for result in result_list]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_collector_nozzle_thermal_v13_markdown(
    results: Iterable[CollectorNozzleThermalV13Result],
    path: Path,
    summary_results: Iterable[CollectorNozzleThermalV13Result] | None = None,
) -> None:
    """Write a collector/nozzle thermal summary."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No collector/nozzle thermal rows to summarize")
    summary_list = (
        _ranked(summary_results) if summary_results is not None else result_list
    )
    if not summary_list:
        raise ValueError("No collector/nozzle thermal summary rows to summarize")
    minimum_ready = next((row for row in summary_list if row.cad_ready), None)
    best_score_ready = max(
        (row for row in summary_list if row.cad_ready),
        key=lambda row: row.cad_readiness_score,
        default=None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v1.3 Collector/Nozzle Thermal Architecture",
        "",
        _summary(minimum_ready, best_score_ready),
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- v1.3 splits rejected heat into bremsstrahlung, transport, collector, nozzle, and power-conditioning channels.",
        "- Only collector/nozzle/conditioning channels are recoverable channel heat in this screen.",
        "- Bremsstrahlung and transport heat remain radiator load.",
        "- A CAD-ready row must close plant-net and keep residual radiator span/area inside constraints.",
        "- CAD-ready here means parametric envelope-ready, not detailed CAD or validated thermal hardware.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_collector_nozzle_thermal_v13(
    results: Iterable[CollectorNozzleThermalV13Result],
    path: Path,
) -> None:
    """Plot channel-specific recovery against radiator span."""
    import matplotlib.pyplot as plt

    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No collector/nozzle thermal rows to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#0f766e" if row.cad_ready else "#b45309" for row in result_list]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        [row.recovered_electric_power_mw for row in result_list],
        [row.adjusted_wing_span_each_m for row in result_list],
        c=colors,
        alpha=0.8,
    )
    ax.axhline(500.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Recovered electric power from recoverable channels (MW)")
    ax.set_ylabel("Adjusted radiator wing span per side (m)")
    ax.set_title("LunarFire v1.3 Collector/Nozzle Thermal Architecture")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ranked(
    results: Iterable[CollectorNozzleThermalV13Result],
) -> List[CollectorNozzleThermalV13Result]:
    return sorted(
        list(results),
        key=lambda row: (
            not row.cad_ready,
            row.recovery_aggressiveness,
            -row.cad_readiness_score,
        ),
    )


def _summary(
    minimum_ready: CollectorNozzleThermalV13Result | None,
    best_score_ready: CollectorNozzleThermalV13Result | None,
) -> str:
    if minimum_ready is None:
        return "CAD-ready collector/nozzle recipe: `none found`."
    assert best_score_ready is not None
    return "\n".join(
        [
            "Lowest heuristic-aggressiveness CAD-ready collector/nozzle recipe:",
            "",
            f"- Plant-net power: `{minimum_ready.plant_net_power_mw:.1f} MW`",
            f"- Collector capture efficiency: `{minimum_ready.collector_capture_efficiency:.2f}`",
            f"- Nozzle capture efficiency: `{minimum_ready.nozzle_capture_efficiency:.2f}`",
            f"- Radiator temperature: `{minimum_ready.radiator_temperature_k:.0f} K`",
            f"- Topology packing factor: `{minimum_ready.topology_packing_factor:.1f}`",
            f"- Recovered electric power: `{minimum_ready.recovered_electric_power_mw:.1f} MW`",
            f"- Recoverable channel heat: `{minimum_ready.recoverable_channel_heat_mw:.1f} MW`",
            f"- Adjusted wing span per side: `{minimum_ready.adjusted_wing_span_each_m:.0f} m`",
            "",
            f"Highest-score CAD-ready row: `{best_score_ready.cad_readiness_score:.3f}` "
            f"at `{best_score_ready.plant_net_power_mw:.1f} MW` plant-net.",
        ]
    )


def _markdown_table(results: List[CollectorNozzleThermalV13Result]) -> str:
    headers = [
        "Ready",
        "Plant MW",
        "Agg",
        "Collector eta",
        "Nozzle eta",
        "Recovered MW",
        "Recoverable MW",
        "Brem MW",
        "Transport MW",
        "Collector MW",
        "Nozzle MW",
        "Rad K",
        "Pack",
        "Span m",
        "Blockers",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                str(result.cad_ready),
                f"{result.plant_net_power_mw:.1f}",
                f"{result.recovery_aggressiveness:.1f}",
                f"{result.collector_capture_efficiency:.2f}",
                f"{result.nozzle_capture_efficiency:.2f}",
                f"{result.recovered_electric_power_mw:.1f}",
                f"{result.recoverable_channel_heat_mw:.1f}",
                f"{result.bremsstrahlung_core_mw:.0f}",
                f"{result.transport_wall_mw:.0f}",
                f"{result.collector_waste_mw:.0f}",
                f"{result.nozzle_waste_mw:.0f}",
                f"{result.radiator_temperature_k:.0f}",
                f"{result.topology_packing_factor:.1f}",
                f"{result.adjusted_wing_span_each_m:.0f}",
                result.blockers,
            ]
        )
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)
