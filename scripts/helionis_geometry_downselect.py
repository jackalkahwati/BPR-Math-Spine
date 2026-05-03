#!/usr/bin/env python3
"""Run the Helionis zero-shot reactor geometry downselect.

Usage:
    python3 scripts/helionis_geometry_downselect.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.architecture import DEFAULT_SCENARIOS
from helionis.geometry import run_geometry_downselect
from helionis.geometry_reporting import (
    plot_frc_field_map,
    plot_geometry_scores,
    write_geometry_csv,
    write_geometry_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Helionis reactor geometry downselect outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/helionis",
        help="Directory for CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = run_geometry_downselect(DEFAULT_SCENARIOS)

    csv_path = output_dir / "helionis_geometry_downselect.csv"
    markdown_path = output_dir / "helionis_geometry_downselect.md"
    score_plot_path = output_dir / "helionis_geometry_scores.png"
    field_plot_path = output_dir / "helionis_frc_field_map.png"

    write_geometry_csv(results, csv_path)
    write_geometry_markdown(results, markdown_path)
    if not args.no_plot:
        plot_geometry_scores(results, score_plot_path)
        plot_frc_field_map(field_plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {score_plot_path}")
        print(f"Wrote {field_plot_path}")


if __name__ == "__main__":
    main()
