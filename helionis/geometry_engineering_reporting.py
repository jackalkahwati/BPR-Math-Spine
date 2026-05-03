"""Reporting helpers for LunarFire plant-net geometry re-score."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.geometry_engineering import GeometryEngineeringResult


def write_geometry_engineering_csv(
    results: Iterable[GeometryEngineeringResult],
    path: Path,
) -> None:
    """Write geometry engineering re-score rows to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No geometry engineering rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_geometry_engineering_markdown(
    results: Iterable[GeometryEngineeringResult],
    path: Path,
) -> None:
    """Write a compact geometry engineering comparison report."""
    result_list = sorted(
        list(results),
        key=_result_sort_key,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No geometry engineering rows to summarize")

    best = result_list[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v0.4 Plant-Net Geometry Re-Score Output",
        "",
        "Model label: `order_of_magnitude_trade_study`.",
        "",
        f"Best current geometry by plant-net power: `{best.family}`.",
        f"Best plant-net power: `{best.plant_net_power_mw:.1f} MW`.",
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- This is a plant-net re-score, not a final geometry selection.",
        "- Profiles use rough geometry-specific assumptions for beta, transport, conversion, and current drive.",
        "- Negative plant-net means the geometry misses under current assumptions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_geometry_engineering(
    results: Iterable[GeometryEngineeringResult],
    path: Path,
) -> None:
    """Plot plant-net comparison by geometry family."""
    import matplotlib.pyplot as plt

    result_list = sorted(
        list(results),
        key=_result_sort_key,
        reverse=True,
    )
    if not result_list:
        raise ValueError("No geometry engineering rows to plot")

    labels = [result.family for result in result_list]
    plant_net = [
        result.plant_net_power_mw if result.feasible_screening_candidate else 0.0
        for result in result_list
    ]
    radiator = [
        result.radiator_area_m2 if result.feasible_screening_candidate else 0.0
        for result in result_list
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_net, ax_radiator) = plt.subplots(2, 1, figsize=(8, 8))
    ax_net.bar(labels, plant_net, color="#334155")
    ax_net.axhline(0.0, color="black", linewidth=0.8)
    ax_net.set_ylabel("Plant net power (MW)")
    ax_net.set_title("LunarFire v0.4 Geometry Engineering Re-Score")

    ax_radiator.bar(labels, radiator, color="#64748b")
    ax_radiator.set_ylabel("Radiator area (m^2)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _markdown_table(results: List[GeometryEngineeringResult]) -> str:
    headers = [
        "Geometry",
        "Feasible",
        "Plant MW",
        "Gross MW",
        "Load MW",
        "R m",
        "L m",
        "B T",
        "Radiator m2",
        "Direct eta",
        "CD frac",
        "Transport",
        "Rejections",
        "Closes",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.family,
                str(result.feasible_screening_candidate),
                _fmt(result.plant_net_power_mw, result.feasible_screening_candidate),
                _fmt(result.gross_fusion_mw, result.feasible_screening_candidate, ".0f"),
                _fmt(result.engineering_load_mw, result.feasible_screening_candidate),
                _fmt(
                    result.separatrix_radius_m,
                    result.feasible_screening_candidate,
                    ".2f",
                ),
                _fmt(result.length_m, result.feasible_screening_candidate, ".2f"),
                _fmt(result.required_field_t, result.feasible_screening_candidate, ".2f"),
                _fmt(
                    result.radiator_area_m2,
                    result.feasible_screening_candidate,
                    ".0f",
                ),
                f"{result.direct_conversion_efficiency:.2f}",
                f"{result.current_drive_fraction_of_gross_fusion:.3f}",
                f"{result.transport_loss_multiplier:.2f}",
                result.rejection_summary or "",
                str(result.closes_engineering_net),
            ]
        )

    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def _result_sort_key(result: GeometryEngineeringResult) -> tuple[bool, float]:
    return (result.feasible_screening_candidate, result.plant_net_power_mw)


def _fmt(value: float, feasible: bool, spec: str = ".1f") -> str:
    if not feasible:
        return "N/A"
    return format(value, spec)
