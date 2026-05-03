"""Reporting helpers for Helionis geometry downselect outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.fields import axisymmetric_frc_field_map
from helionis.geometry import GeometryScore


def write_geometry_csv(results: Iterable[GeometryScore], path: Path) -> None:
    """Write geometry downselect rows to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No geometry rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_geometry_markdown(results: Iterable[GeometryScore], path: Path) -> None:
    """Write a compact geometry downselect memo table."""
    result_list = list(results)
    if not result_list:
        raise ValueError("No geometry rows to summarize")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Helionis Geometry Downselect Output",
        "",
        "Model label: `order_of_magnitude_trade_study`.",
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- Scores are zero-shot architecture fit metrics, not final reactor performance.",
        "- FRC is expected to rank first when compactness and direct charged-particle access matter.",
        "- Magnetic mass is a comparative proxy from magnetic energy and field strength.",
        "- Neutron wall load comes from the existing D-He3 trade-study side-reaction model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_geometry_scores(results: Iterable[GeometryScore], path: Path) -> None:
    """Plot geometry score by scenario and family."""
    import matplotlib.pyplot as plt

    result_list = list(results)
    if not result_list:
        raise ValueError("No geometry rows to plot")

    labels = [
        f"{result.scenario_name.replace('_dhe3', '').replace('_', ' ')}\n{result.family}"
        for result in result_list
    ]
    scores = [result.total_score for result in result_list]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, scores, color="#334155")
    ax.set_ylabel("Geometry fit score")
    ax.set_title("Helionis Zero-Shot Geometry Downselect")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_frc_field_map(path: Path) -> None:
    """Plot the default FRC axisymmetric flux prototype."""
    import matplotlib.pyplot as plt

    field_map = axisymmetric_frc_field_map(
        separatrix_radius_m=1.8,
        length_m=10.0,
        axial_field_t=6.0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    contour = ax.contourf(
        field_map.z_grid,
        field_map.r_grid,
        field_map.flux_webers,
        levels=40,
        cmap="viridis",
    )
    ax.contour(
        field_map.z_grid,
        field_map.r_grid,
        field_map.flux_webers,
        levels=[0.0],
        colors="white",
        linewidths=1.4,
    )
    ax.set_title("FRC Axisymmetric Flux Prototype")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("r (m)")
    fig.colorbar(contour, ax=ax, label="Flux proxy (Wb)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[GeometryScore]) -> str:
    headers = [
        "Scenario",
        "Family",
        "Score",
        "B T",
        "R minor m",
        "L m",
        "Wall MW/m2",
        "Direct",
        "Rationale",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.scenario_name,
                result.family,
                f"{result.total_score:.3f}",
                f"{result.required_field_t:.2f}",
                f"{result.minor_radius_m:.2f}",
                f"{result.length_m:.2f}",
                f"{result.neutron_wall_load_mw_m2:.4f}",
                f"{result.direct_conversion_access:.2f}",
                result.rationale,
            ]
        )

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)
