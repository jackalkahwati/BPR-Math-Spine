"""Reporting helpers for Helionis trade-study outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List

from helionis.architecture import TradeStudyResult


SUMMARY_FIELDS = [
    "name",
    "reaction",
    "temperature_kev",
    "triple_product_kev_s_m3",
    "fusion_power_mw",
    "neutron_fraction_of_fusion_power",
    "bremsstrahlung_loss_mw",
    "useful_power_mw",
    "net_power_mw",
    "required_volume_for_target_m3",
    "shielding_mass_proxy_tonnes",
    "warnings",
]


def write_csv(results: Iterable[TradeStudyResult], path: Path) -> None:
    """Write all trade-study result fields to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No trade-study rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(results: Iterable[TradeStudyResult], path: Path) -> None:
    """Write a compact Markdown summary for partner/investor review."""
    result_list = list(results)
    if not result_list:
        raise ValueError("No trade-study rows to summarize")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Helionis D-He3 Trade Study Output",
        "",
        "Model label: `order_of_magnitude_trade_study`.",
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- `net_power_mw` is useful converted power minus simplified radiation and transport losses.",
        "- `gain_proxy` is not Q-plasma; it is useful converted power divided by modeled losses.",
        "- `required_volume_for_target_m3` is infinite when the net power proxy is negative.",
        "- Shielding mass is a comparative neutron-load proxy, not a mechanical shield design.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trade_study(results: Iterable[TradeStudyResult], path: Path) -> None:
    """Create a compact bar chart for net power and shielding proxy comparison."""
    import matplotlib.pyplot as plt

    result_list = list(results)
    if not result_list:
        raise ValueError("No trade-study rows to plot")

    labels = [result.name.replace("_", "\n") for result in result_list]
    net_power = [result.net_power_mw for result in result_list]
    shielding = [result.shielding_mass_proxy_tonnes for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_power, ax_shielding) = plt.subplots(2, 1, figsize=(11, 8))
    ax_power.bar(labels, net_power, color="#334155")
    ax_power.axhline(0.0, color="black", linewidth=0.8)
    ax_power.set_ylabel("Net power proxy (MW)")
    ax_power.set_title("Helionis Screening Scenarios")

    ax_shielding.bar(labels, shielding, color="#64748b")
    ax_shielding.set_ylabel("Shielding proxy (tonnes)")
    ax_shielding.tick_params(axis="x", labelrotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[TradeStudyResult]) -> str:
    headers = [
        "Scenario",
        "Fuel",
        "T keV",
        "Fusion MW",
        "Neutron %",
        "Useful MW",
        "Net MW",
        "Shield t",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.name,
                result.reaction,
                f"{result.temperature_kev:.0f}",
                f"{result.fusion_power_mw:.1f}",
                f"{100.0 * result.neutron_fraction_of_fusion_power:.2f}",
                f"{result.useful_power_mw:.1f}",
                f"{result.net_power_mw:.1f}",
                _format_float(result.shielding_mass_proxy_tonnes),
            ]
        )

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.1f}"
