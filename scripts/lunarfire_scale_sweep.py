#!/usr/bin/env python3
"""Generate the LunarFire v0.3 minimum viable scale sweep.

Usage:
    python3 scripts/lunarfire_scale_sweep.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.scale_reporting import plot_scale_sweep, write_scale_csv, write_scale_markdown
from helionis.scale_sweep import DEFAULT_TARGETS_MW, run_scale_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire minimum viable scale outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/helionis",
        help="Directory for CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGETS_MW),
        help="Screening-net target powers in MW.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = run_scale_sweep(targets_mw=tuple(args.targets))

    csv_path = output_dir / "lunarfire_scale_sweep_v0_3.csv"
    markdown_path = output_dir / "lunarfire_scale_sweep_v0_3.md"
    plot_path = output_dir / "lunarfire_scale_sweep_v0_3.png"

    write_scale_csv(results, csv_path)
    write_scale_markdown(results, markdown_path)
    if not args.no_plot:
        plot_scale_sweep(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
