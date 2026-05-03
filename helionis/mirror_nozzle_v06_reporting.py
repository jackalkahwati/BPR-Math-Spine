"""Reporting helpers for LunarFire v0.6 mirror/nozzle sweep."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.mirror_nozzle_v06 import MirrorNozzleV06Result


def write_mirror_nozzle_v06_csv(
    results: Iterable[MirrorNozzleV06Result],
    path: Path,
) -> None:
    """Write mirror/nozzle v0.6 sweep rows to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No mirror/nozzle v0.6 rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_mirror_nozzle_v06_markdown(
    results: Iterable[MirrorNozzleV06Result],
    path: Path,
) -> None:
    """Write a compact mirror/nozzle v0.6 summary."""
    result_list = sorted(
        list(results),
        key=lambda result: result.plant_net_power_mw,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No mirror/nozzle v0.6 rows to summarize")

    best = result_list[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v0.6 Mirror/Nozzle Output",
        "",
        f"Best plant-net power: `{best.plant_net_power_mw:.1f} MW`.",
        f"Best mirror ratio: `{best.mirror_ratio:.1f}`.",
        f"Top-row loss cone fraction: `{best.loss_cone_fraction:.3f}`.",
        f"Top-row alpha collector: `{best.alpha_collector_voltage_kv:.0f} kV`.",
        f"Top-row proton collector: `{best.proton_collector_voltage_kv:.0f} kV`.",
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
        "- Loss cone fraction is an isotropic two-ended mirror proxy.",
        "- Alpha and proton collector stages are matched separately.",
        "- Collector area is sized from incident charged-particle power and an assumed power density cap.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_mirror_nozzle_v06(
    results: Iterable[MirrorNozzleV06Result],
    path: Path,
) -> None:
    """Plot v0.6 plant-net and plug-field burden."""
    import matplotlib.pyplot as plt

    result_list = sorted(
        list(results),
        key=lambda result: result.plant_net_power_mw,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No mirror/nozzle v0.6 rows to plot")

    labels = [
        f"R{result.mirror_ratio:.1f}\nT{result.temperature_kev:.0f}/tau{result.confinement_s:.0f}"
        for result in result_list
    ]
    plant_net = [result.plant_net_power_mw for result in result_list]
    plug_fields = [result.plug_field_t for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_net, ax_field) = plt.subplots(2, 1, figsize=(10, 8))
    ax_net.bar(labels, plant_net, color="#334155")
    ax_net.axhline(0.0, color="black", linewidth=0.8)
    ax_net.set_ylabel("Plant net power (MW)")
    ax_net.set_title("LunarFire v0.6 Mirror/Nozzle Sweep")

    ax_field.bar(labels, plug_fields, color="#64748b")
    ax_field.set_ylabel("Plug field (T)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[MirrorNozzleV06Result]) -> str:
    headers = [
        "Plant MW",
        "T keV",
        "n m^-3",
        "tau s",
        "Mirror ratio",
        "Loss cone",
        "LC scale",
        "Transport",
        "Midplane T",
        "Plug T",
        "Alpha kV",
        "Proton kV",
        "Direct eta",
        "Collector m2",
        "Load MW",
        "Reject MW",
        "Plug mass t",
        "Eff magnet t",
        "Gross MW",
        "Radiator m2",
        "Closes",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                f"{result.plant_net_power_mw:.1f}",
                f"{result.temperature_kev:.0f}",
                f"{result.ion_density_m3:.1e}",
                f"{result.confinement_s:.0f}",
                f"{result.mirror_ratio:.1f}",
                f"{result.loss_cone_fraction:.3f}",
                f"{result.loss_cone_transport_scale:.1f}",
                f"{result.transport_loss_multiplier:.2f}",
                f"{result.midplane_field_t:.1f}",
                f"{result.plug_field_t:.1f}",
                f"{result.alpha_collector_voltage_kv:.0f}",
                f"{result.proton_collector_voltage_kv:.0f}",
                f"{result.direct_conversion_efficiency:.2f}",
                f"{result.collector_area_m2:.0f}",
                f"{result.engineering_load_mw:.1f}",
                f"{result.rejected_heat_mw:.0f}",
                f"{result.plug_coil_mass_proxy_tonnes:.0f}",
                f"{result.effective_magnet_mass_proxy_tonnes:.0f}",
                f"{result.gross_fusion_mw:.0f}",
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
