"""Reporting helpers for LunarFire v0.7 architecture comparison."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.architecture_comparison_v07 import ArchitectureV07Result


def write_architecture_v07_csv(
    results: Iterable[ArchitectureV07Result],
    path: Path,
) -> None:
    """Write v0.7 architecture comparison rows to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No v0.7 architecture rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_architecture_v07_markdown(
    results: Iterable[ArchitectureV07Result],
    path: Path,
) -> None:
    """Write same-assumption architecture comparison summary."""
    result_list = sorted(
        list(results),
        key=lambda result: result.plant_net_power_mw,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No v0.7 architecture rows to summarize")

    best = result_list[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v0.7 Shared-Grid Architecture Output",
        "",
        "This is a shared-grid/shared-accounting FRC vs mirror/nozzle comparison.",
        f"Top plant-net architecture: `{best.family}`.",
        f"Top plant-net power: `{best.plant_net_power_mw:.1f} MW`.",
        (
            "Plant-net status: `closes`."
            if best.closes_engineering_net
            else "Plant-net status: `does not close`."
        ),
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- Both rows use the same target, plasma grid, thermal conversion, and engineering-net path.",
        "- Direct-conversion and transport assumptions remain architecture-specific.",
        "- Mirror/nozzle uses a pitch-angle-scattering leakage proxy.",
        "- Mirror/nozzle includes a first-order collector/nozzle auxiliary load.",
        "- FRC remains a baseline row here, not a high-fidelity FRC physics model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_architecture_v07(
    results: Iterable[ArchitectureV07Result],
    path: Path,
) -> None:
    """Plot plant-net comparison for the v0.7 architecture rows."""
    import matplotlib.pyplot as plt

    result_list = sorted(
        list(results),
        key=lambda result: result.plant_net_power_mw,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No v0.7 architecture rows to plot")

    labels = [result.family for result in result_list]
    plant_net = [result.plant_net_power_mw for result in result_list]
    rejected_heat = [result.rejected_heat_mw for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_net, ax_heat) = plt.subplots(2, 1, figsize=(8, 8))
    ax_net.bar(labels, plant_net, color="#334155")
    ax_net.axhline(0.0, color="black", linewidth=0.8)
    ax_net.set_ylabel("Plant net power (MW)")
    ax_net.set_title("LunarFire v0.7 Same-Assumption Comparison")

    ax_heat.bar(labels, rejected_heat, color="#64748b")
    ax_heat.set_ylabel("Rejected heat (MW)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[ArchitectureV07Result]) -> str:
    headers = [
        "Family",
        "Plant MW",
        "Closes",
        "Gross MW",
        "Load MW",
        "Collector MW",
        "Reject MW",
        "Transport",
        "Direct eta",
        "R m",
        "L m",
        "B T",
        "Plug T",
        "Radiator m2",
        "Warnings",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.family,
                f"{result.plant_net_power_mw:.1f}",
                str(result.closes_engineering_net),
                f"{result.gross_fusion_mw:.0f}",
                f"{result.engineering_load_mw:.1f}",
                f"{result.collector_nozzle_load_mw:.1f}",
                f"{result.rejected_heat_mw:.0f}",
                f"{result.transport_loss_multiplier:.2f}",
                f"{result.direct_conversion_efficiency:.2f}",
                f"{result.separatrix_radius_m:.2f}",
                f"{result.length_m:.2f}",
                f"{result.midplane_field_t:.1f}",
                _fmt_optional(result.plug_field_t, result.family == "mirror_nozzle"),
                f"{result.radiator_area_m2:.0f}",
                result.warnings,
            ]
        )

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def _fmt_optional(value: float, present: bool) -> str:
    return f"{value:.1f}" if present else "N/A"
