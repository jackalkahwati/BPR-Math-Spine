#!/usr/bin/env python3
"""Generate the LunarFire v0.4 plant-net geometry re-score.

Usage:
    python3 scripts/lunarfire_geometry_engineering.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.geometry_engineering import run_geometry_engineering_rescore
from helionis.geometry_engineering_reporting import (
    plot_geometry_engineering,
    write_geometry_engineering_csv,
    write_geometry_engineering_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire plant-net geometry re-score outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/helionis",
        help="Directory for CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--target-mw",
        type=float,
        default=50.0,
        help="Screening-net target power in MW.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = run_geometry_engineering_rescore(
        target_screening_net_mw=args.target_mw,
    )

    csv_path = output_dir / "lunarfire_geometry_engineering_v0_4.csv"
    markdown_path = output_dir / "lunarfire_geometry_engineering_v0_4.md"
    plot_path = output_dir / "lunarfire_geometry_engineering_v0_4.png"

    write_geometry_engineering_csv(results, csv_path)
    write_geometry_engineering_markdown(results, markdown_path)
    if not args.no_plot:
        plot_geometry_engineering(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
