"""Reporting helpers for LunarFire v0.8 margin recovery."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from helionis.margin_recovery_v08 import MarginRecoveryV08Result


def write_margin_recovery_v08_csv(
    results: Iterable[MarginRecoveryV08Result],
    path: Path,
) -> None:
    """Write v0.8 margin recovery rows to CSV."""
    rows = [result.to_row() for result in results]
    if not rows:
        raise ValueError("No v0.8 margin recovery rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_margin_recovery_v08_markdown(
    results: Iterable[MarginRecoveryV08Result],
    path: Path,
) -> None:
    """Write a v0.8 margin recovery summary."""
    result_list = sorted(
        list(results),
        key=_rank_key,
    )
    if not result_list:
        raise ValueError("No v0.8 margin recovery rows to summarize")

    recipe = _minimum_from_rows(result_list)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LunarFire v0.8 Margin Recovery Output",
        "",
        _recipe_summary(recipe),
        "",
        _markdown_table(result_list),
        "",
        "Interpretation notes:",
        "",
        "- The minimum recovery recipe is the least aggressive closing row in the sweep.",
        "- Aggressiveness is a relative heuristic, not a physics probability.",
        "- Closing rows still depend on screening proxies for confinement and direct conversion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_margin_recovery_v08(
    results: Iterable[MarginRecoveryV08Result],
    path: Path,
) -> None:
    """Plot plant-net margin versus recovery aggressiveness."""
    import matplotlib.pyplot as plt

    result_list = list(results)
    if not result_list:
        raise ValueError("No v0.8 margin recovery rows to plot")

    colors = ["#0f766e" if row.closes_engineering_net else "#64748b" for row in result_list]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        [row.aggressiveness_score for row in result_list],
        [row.plant_net_power_mw for row in result_list],
        c=colors,
        alpha=0.8,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Recovery aggressiveness score")
    ax.set_ylabel("Plant net power (MW)")
    ax.set_title("LunarFire v0.8 Mirror/Nozzle Margin Recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _minimum_from_rows(
    rows: List[MarginRecoveryV08Result],
) -> MarginRecoveryV08Result | None:
    closing_rows = [row for row in rows if row.closes_engineering_net]
    if not closing_rows:
        return None
    return min(
        closing_rows,
        key=lambda row: (row.aggressiveness_score, -row.plant_net_power_mw),
    )


def _recipe_summary(recipe: MarginRecoveryV08Result | None) -> str:
    if recipe is None:
        return "Minimum recovery recipe: `none found`."
    return "\n".join(
        [
            "Minimum recovery recipe:",
            "",
            f"- Plant-net: `{recipe.plant_net_power_mw:.1f} MW`",
            f"- Aggressiveness score: `{recipe.aggressiveness_score:.1f}`",
            f"- Pitch-angle scattering time: `{recipe.pitch_angle_scattering_s:.0f} s`",
            f"- Mirror stabilization factor: `{recipe.mirror_stabilization_factor:.1f}`",
            f"- Direct conversion cap: `{recipe.direct_conversion_cap:.2f}`",
            f"- Collector base efficiency: `{recipe.collector_base_efficiency:.2f}`",
            f"- Collector match bonus: `{recipe.collector_match_bonus:.2f}`",
            f"- Plug-coil mass coefficient: `{recipe.plug_coil_mass_coefficient_tonnes_per_t2:.2f}`",
            f"- Collector auxiliary load: `{recipe.collector_aux_kw_per_m2:.1f} kW/m2`",
            f"- Nozzle auxiliary fraction: `{recipe.nozzle_aux_fraction_of_direct_power:.3f}`",
        ]
    )


def _markdown_table(results: List[MarginRecoveryV08Result]) -> str:
    headers = [
        "Plant MW",
        "Closes",
        "Score",
        "Scatter s",
        "Stabilize",
        "Direct cap",
        "Base eta",
        "Match bonus",
        "Direct eta",
        "Cap limited",
        "Plug coeff",
        "Collector kW/m2",
        "Nozzle frac",
        "Transport",
        "Gross MW",
        "Radiator m2",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                f"{result.plant_net_power_mw:.1f}",
                str(result.closes_engineering_net),
                f"{result.aggressiveness_score:.1f}",
                f"{result.pitch_angle_scattering_s:.0f}",
                f"{result.mirror_stabilization_factor:.1f}",
                f"{result.direct_conversion_cap:.2f}",
                f"{result.collector_base_efficiency:.2f}",
                f"{result.collector_match_bonus:.2f}",
                f"{result.direct_conversion_efficiency:.2f}",
                str(result.is_direct_conversion_cap_limited),
                f"{result.plug_coil_mass_coefficient_tonnes_per_t2:.2f}",
                f"{result.collector_aux_kw_per_m2:.1f}",
                f"{result.nozzle_aux_fraction_of_direct_power:.3f}",
                f"{result.transport_loss_multiplier:.3f}",
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


def _rank_key(row: MarginRecoveryV08Result) -> tuple[bool, float, float]:
    return (not row.closes_engineering_net, row.aggressiveness_score, -row.plant_net_power_mw)
