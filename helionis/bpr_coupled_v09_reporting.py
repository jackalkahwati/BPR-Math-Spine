"""Reporting helpers for LunarFire v0.9 BPR-coupled screen."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.bpr_coupled_v09 import BPRCoupledV09Result


def write_bpr_coupled_v09_csv(
    results: Iterable[BPRCoupledV09Result],
    path: Path,
) -> None:
    """Write v0.9 BPR-coupled rows to CSV."""
    result_list = sorted(
        list(results),
        key=lambda row: row.plant_net_power_mw,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No v0.9 BPR-coupled rows to write")
    rows = [result.to_row() for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_bpr_coupled_v09_markdown(
    results: Iterable[BPRCoupledV09Result],
    path: Path,
) -> None:
    """Write a v0.9 BPR-coupled summary."""
    result_list = sorted(
        list(results),
        key=lambda row: row.plant_net_power_mw,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No v0.9 BPR-coupled rows to summarize")

    best = result_list[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v0.9 BPR-Coupled Output",
        "",
        f"Best plant-net power: `{best.plant_net_power_mw:.1f} MW`.",
        f"BPR sources: `{best.bpr_source_modules}`.",
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- v0.9 uses existing BPR Math Spine primitives as bounded correction factors.",
        "- BPR factors are not treated as energy sources.",
        "- Current BPR source modules: `bpr.impedance` and `bpr.resonance`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bpr_coupled_v09(
    results: Iterable[BPRCoupledV09Result],
    path: Path,
) -> None:
    """Plot v0.9 BPR-coupled plant-net results."""
    import matplotlib.pyplot as plt

    result_list = sorted(
        list(results),
        key=lambda row: row.plant_net_power_mw,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No v0.9 BPR-coupled rows to plot")

    labels = [
        f"#{idx} R{row.mirror_ratio:.1f} A{row.mirror_aspect_ratio:.0f}\n"
        f"n{row.ion_density_m3:.1e} tau{row.confinement_s:.0f}"
        for idx, row in enumerate(result_list, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, [row.plant_net_power_mw for row in result_list], color="#334155")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Plant net power (MW)")
    ax.set_title("LunarFire v0.9 BPR-Coupled Mirror/Nozzle Screen")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[BPRCoupledV09Result]) -> str:
    headers = [
        "Plant MW",
        "Closes",
        "R",
        "Aspect",
        "T keV",
        "n m^-3",
        "tau s",
        "BPR align",
        "BPR Z match",
        "BPR transport x",
        "BPR direct x",
        "Gross MW",
        "Radiator m2",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                f"{result.plant_net_power_mw:.1f}",
                str(result.closes_engineering_net),
                f"{result.mirror_ratio:.1f}",
                f"{result.mirror_aspect_ratio:.0f}",
                f"{result.temperature_kev:.0f}",
                f"{result.ion_density_m3:.1e}",
                f"{result.confinement_s:.0f}",
                f"{result.bpr_resonance_alignment:.3f}",
                f"{result.bpr_impedance_match:.3f}",
                f"{result.bpr_transport_multiplier:.3f}",
                f"{result.bpr_direct_conversion_multiplier:.3f}",
                f"{result.gross_fusion_mw:.0f}",
                f"{result.radiator_area_m2:.0f}",
            ]
        )

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)
