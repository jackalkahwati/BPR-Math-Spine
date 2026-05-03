"""Reporting helpers for LunarFire scale-sweep outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.scale_sweep import ScaleSweepResult


def write_scale_csv(results: Iterable[ScaleSweepResult], path: Path) -> None:
    """Write scale-sweep rows to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No scale-sweep rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_scale_markdown(
    results: Iterable[ScaleSweepResult],
    path: Path,
) -> None:
    """Write a compact minimum-viable-scale summary."""
    result_list = list(results)
    if not result_list:
        raise ValueError("No scale-sweep rows to summarize")
    result_list = sorted(result_list, key=lambda result: result.target_screening_net_mw)
    first_closed = next(
        (result for result in result_list if result.closes_engineering_net),
        None,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v0.3 Minimum Viable Scale Output",
        "",
        "Model label: `order_of_magnitude_trade_study`.",
        "",
        _summary(first_closed),
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- Targets are screening-net power levels, not guaranteed delivered plant output.",
        "- Plant-net power subtracts first-order engineering loads from each target case.",
        "- Positive plant-net means the target closes under the current rough assumptions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scale_sweep(results: Iterable[ScaleSweepResult], path: Path) -> None:
    """Plot plant-net power across target scale."""
    import matplotlib.pyplot as plt

    result_list = list(results)
    if not result_list:
        raise ValueError("No scale-sweep rows to plot")

    targets = [result.target_screening_net_mw for result in result_list]
    plant_net = [result.plant_net_power_mw for result in result_list]
    radiator_area = [result.radiator_area_m2 for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_power, ax_radiator) = plt.subplots(2, 1, figsize=(9, 8))
    ax_power.plot(targets, plant_net, marker="o", color="#334155")
    ax_power.axhline(0.0, color="black", linewidth=0.8)
    ax_power.set_xscale("log")
    ax_power.set_ylabel("Plant net power (MW)")
    ax_power.set_title("LunarFire Minimum Viable Scale Sweep")

    ax_radiator.plot(targets, radiator_area, marker="o", color="#64748b")
    ax_radiator.set_xscale("log")
    ax_radiator.set_xlabel("Target screening-net power (MW)")
    ax_radiator.set_ylabel("Radiator area (m^2)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _summary(first_closed: ScaleSweepResult | None) -> str:
    if first_closed is None:
        return "No tested target closes plant-net power under current assumptions."
    return (
        "First tested target that closes plant-net power: "
        f"`{first_closed.target_screening_net_mw:.0f} MW` screening-net, "
        f"yielding `{first_closed.plant_net_power_mw:.1f} MW` plant-net."
    )


def _markdown_table(results: List[ScaleSweepResult]) -> str:
    headers = [
        "Target MW",
        "Plant MW",
        "Margin %",
        "Gross MW",
        "Load MW",
        "R m",
        "L m",
        "B T",
        "Radiator m2",
        "Closes",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                f"{result.target_screening_net_mw:.0f}",
                f"{result.plant_net_power_mw:.1f}",
                f"{100.0 * result.plant_net_margin_fraction:.1f}",
                f"{result.gross_fusion_mw:.0f}",
                f"{result.engineering_load_mw:.1f}",
                f"{result.separatrix_radius_m:.2f}",
                f"{result.length_m:.2f}",
                f"{result.required_field_t:.2f}",
                f"{result.radiator_area_m2:.0f}",
                str(result.closes_engineering_net),
            ]
        )

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)
