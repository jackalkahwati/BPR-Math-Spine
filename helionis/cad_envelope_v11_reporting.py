"""Reporting helpers for LunarFire v1.1 CAD envelope outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.cad_envelope_v11 import CADEnvelopeV11Result


def write_cad_envelope_v11_csv(
    results: Iterable[CADEnvelopeV11Result],
    path: Path,
) -> None:
    """Write CAD envelope rows to CSV."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No CAD envelope rows to write")
    rows = [result.to_row() for result in result_list]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_cad_envelope_v11_markdown(
    results: Iterable[CADEnvelopeV11Result],
    path: Path,
) -> None:
    """Write a CAD envelope summary."""
    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No CAD envelope rows to summarize")
    best = result_list[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v1.1 Parametric CAD Envelope",
        "",
        f"Best CAD-readiness score: `{best.cad_readiness_score:.3f}`.",
        f"Best row CAD-ready: `{best.cad_ready}`.",
        "",
        _readiness_blockers(result_list),
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- v1.1 is a control-constrained parametric envelope, not detailed CAD.",
        "- Dimensions are derived from Helionis v0.9 geometry and Modulus Fusion v1.0 control rows.",
        "- The collector, nozzle, coil, and radiator envelopes are sizing constraints for the next CAD pass.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cad_envelope_v11(
    results: Iterable[CADEnvelopeV11Result],
    path: Path,
) -> None:
    """Plot CAD readiness against machine length and outer radius."""
    import matplotlib.pyplot as plt

    result_list = _ranked(results)
    if not result_list:
        raise ValueError("No CAD envelope rows to plot")

    labels = [f"#{idx}" for idx, _ in enumerate(result_list, start=1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        [row.machine_length_m for row in result_list],
        [row.outer_radius_m for row in result_list],
        s=[60 + 220 * row.cad_readiness_score for row in result_list],
        c=["#0f766e" if row.cad_ready else "#b45309" for row in result_list],
        alpha=0.85,
    )
    for label, row in zip(labels, result_list):
        ax.annotate(label, (row.machine_length_m, row.outer_radius_m), fontsize=8)
    ax.set_xlabel("Machine length (m)")
    ax.set_ylabel("Outer radius (m)")
    ax.set_title("LunarFire v1.1 Control-Constrained CAD Envelope")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ranked(results: Iterable[CADEnvelopeV11Result]) -> List[CADEnvelopeV11Result]:
    return sorted(
        list(results),
        key=lambda row: (row.cad_ready, row.cad_readiness_score),
        reverse=True,
    )


def _markdown_table(results: List[CADEnvelopeV11Result]) -> str:
    headers = [
        "CAD score",
        "Ready",
        "Length m",
        "Outer R m",
        "Plasma R m",
        "Nozzle m",
        "Collector m2",
        "Radiator m2",
        "Wing span m",
        "Control score",
        "Plant MW",
        "Blockers",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                f"{result.cad_readiness_score:.3f}",
                str(result.cad_ready),
                f"{result.machine_length_m:.1f}",
                f"{result.outer_radius_m:.1f}",
                f"{result.plasma_radius_m:.2f}",
                f"{result.nozzle_length_m:.1f}",
                f"{result.collector_surface_area_m2:.0f}",
                f"{result.radiator_area_m2:.0f}",
                f"{result.radiator_wing_span_each_m:.0f}",
                f"{result.source_controllability_score:.3f}",
                f"{result.source_plant_net_power_mw:.1f}",
                result.blockers,
            ]
        )
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def _readiness_blockers(results: List[CADEnvelopeV11Result]) -> str:
    ready_count = sum(1 for row in results if row.cad_ready)
    if ready_count:
        return f"CAD-ready rows: `{ready_count}`."
    blocker_counts = {}
    for row in results:
        for blocker in row.blockers.split("; "):
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    blocker_text = ", ".join(
        f"`{name}` ({count})" for name, count in sorted(blocker_counts.items())
    )
    return f"No CAD-ready rows. Readiness blockers: {blocker_text}."
