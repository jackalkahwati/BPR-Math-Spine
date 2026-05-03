"""Reporting helpers for LunarFire reference-design outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.reference_design import ReferenceDesignResult


def write_reference_csv(results: Iterable[ReferenceDesignResult], path: Path) -> None:
    """Write LunarFire reference-design candidates to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No reference-design rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reference_markdown(
    results: Iterable[ReferenceDesignResult],
    path: Path,
) -> None:
    """Write a compact LunarFire 10 MW reference-design summary."""
    result_list = list(results)
    if not result_list:
        raise ValueError("No reference-design rows to summarize")
    result_list = sorted(result_list, key=lambda result: result.objective_score)

    best = result_list[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire 10 MW FRC Reference Design Output",
        "",
        "Model label: `order_of_magnitude_trade_study`.",
        "",
        "Top-ranked candidate:",
        "",
        f"- Screening net power: {best.screening_net_power_mw:.2f} MW",
        f"- Gross fusion power: {best.gross_fusion_mw:.2f} MW",
        f"- Useful converted power before modeled losses: {best.useful_power_mw:.2f} MW",
        f"- Bremsstrahlung loss: {best.bremsstrahlung_loss_mw:.2f} MW",
        f"- Transport loss: {best.transport_loss_mw:.2f} MW",
        f"- Gain proxy: {best.gain_proxy:.3f}",
        f"- Temperature: {best.temperature_kev:.0f} keV",
        f"- Ion density: {best.ion_density_m3:.2e} m^-3",
        f"- Confinement time: {best.confinement_s:.1f} s",
        f"- FRC radius: {best.separatrix_radius_m:.2f} m",
        f"- FRC length: {best.length_m:.2f} m",
        f"- Required field: {best.required_field_t:.2f} T",
        f"- Neutron wall load: {best.neutron_wall_load_mw_m2:.4f} MW/m^2",
        f"- Warnings: {best.warnings}",
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- This is a reference-design screen, not an engineering drawing.",
        "- Feasible rows close a 10 MW screening-net target under current optimistic assumptions.",
        "- Screening net power excludes recirculating power, cryogenics, current drive, formation power, thermal rejection, and fixed plant loads.",
        "- The assumptions should be stressed next with confinement, recirculating-power, and magnet models.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_reference_candidates(
    results: Iterable[ReferenceDesignResult],
    path: Path,
) -> None:
    """Plot candidate volume and field for the LunarFire reference design."""
    import matplotlib.pyplot as plt

    result_list = list(results)
    if not result_list:
        raise ValueError("No reference-design rows to plot")

    labels = [
        f"{result.temperature_kev:.0f}keV\n{result.ion_density_m3 / 1e20:.1f}e20\n{result.confinement_s:.0f}s"
        for result in result_list
    ]
    volumes = [result.plasma_volume_m3 for result in result_list]
    fields = [result.required_field_t for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_volume, ax_field) = plt.subplots(2, 1, figsize=(11, 8))
    ax_volume.bar(labels, volumes, color="#334155")
    ax_volume.set_ylabel("Plasma volume (m^3)")
    ax_volume.set_title("LunarFire 10 MW FRC Candidate Window")

    ax_field.bar(labels, fields, color="#64748b")
    ax_field.set_ylabel("Required field (T)")
    ax_field.tick_params(axis="x", labelrotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[ReferenceDesignResult]) -> str:
    headers = [
        "Rank",
        "T keV",
        "n 1e20",
        "tau s",
        "R m",
        "L m",
        "B T",
        "Fusion MW",
        "Net MW",
        "Gain",
        "Wall MW/m2",
    ]
    rows = []
    for idx, result in enumerate(results, start=1):
        rows.append(
            [
                str(idx),
                f"{result.temperature_kev:.0f}",
                f"{result.ion_density_m3 / 1.0e20:.1f}",
                f"{result.confinement_s:.0f}",
                f"{result.separatrix_radius_m:.2f}",
                f"{result.length_m:.2f}",
                f"{result.required_field_t:.2f}",
                f"{result.gross_fusion_mw:.1f}",
                f"{result.screening_net_power_mw:.1f}",
                f"{result.gain_proxy:.3f}",
                f"{result.neutron_wall_load_mw_m2:.4f}",
            ]
        )

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)
