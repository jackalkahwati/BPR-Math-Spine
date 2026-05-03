"""Reporting helpers for LunarFire engineering net-power outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.engineering_net import EngineeringNetResult


def write_engineering_csv(
    results: Iterable[EngineeringNetResult],
    path: Path,
) -> None:
    """Write engineering net-power rows to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No engineering net-power rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_engineering_markdown(
    results: Iterable[EngineeringNetResult],
    path: Path,
) -> None:
    """Write the LunarFire v0.2 engineering net-power summary."""
    result_list = list(results)
    if not result_list:
        raise ValueError("No engineering net-power rows to summarize")
    result_list = sorted(
        result_list,
        key=lambda result: result.plant_net_power_mw,
        reverse=True,
    )
    best = result_list[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v0.2 Engineering Net-Power Output",
        "",
        "Model label: `order_of_magnitude_trade_study`.",
        "",
        "Top-ranked engineering case:",
        "",
        f"- Screening net power: {best.screening_net_power_mw:.2f} MW",
        f"- Engineering load: {best.engineering_load_mw:.2f} MW",
        f"- Plant net power: {best.plant_net_power_mw:.2f} MW",
        f"- Closes plant net: {best.closes_engineering_net}",
        f"- Current drive: {best.current_drive_mw:.2f} MW",
        f"- Formation average: {best.formation_average_mw:.2f} MW",
        f"- Cryogenic wall power: {best.cryogenic_wallplug_mw:.2f} MW",
        f"- Power conditioning loss: {best.power_conditioning_loss_mw:.2f} MW",
        f"- Fusion conversion waste heat: {best.conversion_waste_mw:.2f} MW",
        f"- Thermal rejection parasitic: {best.thermal_rejection_parasitic_mw:.2f} MW",
        f"- Rejected heat: {best.rejected_heat_mw:.2f} MW",
        f"- Radiator area: {best.radiator_area_m2:.0f} m^2",
        f"- Warnings: {best.warnings}",
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- Plant-net power subtracts first-order engineering loads from screening-net power.",
        "- This still omits detailed coil stress, quench protection, wall design, and full balance-of-plant engineering.",
        "- Negative plant-net power means the reference point misses after internal reactor loads.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_engineering_net(
    results: Iterable[EngineeringNetResult],
    path: Path,
) -> None:
    """Plot plant-net power and engineering loads for candidate comparison."""
    import matplotlib.pyplot as plt

    result_list = list(results)
    if not result_list:
        raise ValueError("No engineering net-power rows to plot")

    labels = [
        (
            f"{result.temperature_kev:.0f}keV\n"
            f"{result.ion_density_m3 / 1e20:.1f}e20\n"
            f"{result.confinement_s:.0f}s"
        )
        for result in result_list
    ]
    plant_net = [result.plant_net_power_mw for result in result_list]
    loads = [result.engineering_load_mw for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_net, ax_load) = plt.subplots(2, 1, figsize=(11, 8))
    ax_net.bar(labels, plant_net, color="#334155")
    ax_net.axhline(0.0, color="black", linewidth=0.8)
    ax_net.set_ylabel("Plant net power (MW)")
    ax_net.set_title("LunarFire v0.2 Engineering Net-Power Budget")

    ax_load.bar(labels, loads, color="#64748b")
    ax_load.set_ylabel("Engineering load (MW)")
    ax_load.tick_params(axis="x", labelrotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[EngineeringNetResult]) -> str:
    headers = [
        "Rank",
        "Plant MW",
        "Load MW",
        "Current MW",
        "Cond MW",
        "Cryo MW",
        "Reject MW",
        "Radiator m2",
        "Closes",
    ]
    rows = []
    for idx, result in enumerate(results, start=1):
        rows.append(
            [
                str(idx),
                f"{result.plant_net_power_mw:.1f}",
                f"{result.engineering_load_mw:.1f}",
                f"{result.current_drive_mw:.1f}",
                f"{result.power_conditioning_loss_mw:.1f}",
                f"{result.cryogenic_wallplug_mw:.1f}",
                f"{result.rejected_heat_mw:.1f}",
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
