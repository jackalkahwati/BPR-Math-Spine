"""Reporting helpers for the Modulus Fusion control twin."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.modulus_fusion_control import ModulusFusionControlResult


def write_modulus_fusion_control_csv(
    results: Iterable[ModulusFusionControlResult],
    path: Path,
) -> None:
    """Write Modulus Fusion control-twin rows to CSV."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No Modulus Fusion control rows to write")
    rows = [result.to_row() for result in result_list]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_modulus_fusion_control_markdown(
    results: Iterable[ModulusFusionControlResult],
    path: Path,
) -> None:
    """Write a Modulus Fusion control-twin summary."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No Modulus Fusion control rows to summarize")
    best = result_list[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Modulus Fusion Control Twin",
        "",
        f"Best ranked-row score: `{best.controllability_score:.3f}`.",
        f"Best row controllable: `{best.controllable}`.",
        f"Drift claim: `{best.drift_claim}`.",
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- `zero numerical drift` means the deterministic control math adds no roundoff drift.",
        "- This is not zero plasma motion, zero sensor noise, or zero hardware latency.",
        "- The current twin is a control-screening model, not a validated MHD controller.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_modulus_fusion_control(
    results: Iterable[ModulusFusionControlResult],
    path: Path,
) -> None:
    """Plot Modulus Fusion controllability scores."""
    import matplotlib.pyplot as plt

    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No Modulus Fusion control rows to plot")

    labels = [
        f"#{idx} R{row.mirror_ratio:.1f} A{row.mirror_aspect_ratio:.0f}\n"
        f"{row.update_period_ms:.1f} ms"
        for idx, row in enumerate(result_list, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, [row.controllability_score for row in result_list], color="#0f766e")
    ax.axhline(0.75, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Controllability score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Modulus Fusion v1.0 Control Twin")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ranked(
    results: Iterable[ModulusFusionControlResult],
) -> List[ModulusFusionControlResult]:
    return sorted(
        list(results),
        key=lambda row: (row.controllable, row.controllability_score),
        reverse=True,
    )


def _markdown_table(results: List[ModulusFusionControlResult]) -> str:
    headers = [
        "Score",
        "Controllable",
        "Update ms",
        "Coil cmd",
        "Eq residual",
        "Numerical drift",
        "Physical drift",
        "Sensor error",
        "R",
        "Aspect",
        "Plant MW",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                f"{result.controllability_score:.3f}",
                str(result.controllable),
                f"{result.update_period_ms:.2f}",
                f"{result.coil_command_fraction:.3f}",
                f"{result.equilibrium_residual:.3f}",
                f"{result.numerical_drift_fraction:.1e}",
                f"{result.physical_drift_fraction:.3e}",
                f"{result.sensor_error_fraction:.3e}",
                f"{result.mirror_ratio:.1f}",
                f"{result.mirror_aspect_ratio:.0f}",
                f"{result.source_plant_net_power_mw:.1f}",
            ]
        )
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)
