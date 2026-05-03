#!/usr/bin/env python3
"""Generate the LunarFire v0.7 same-assumption architecture comparison.

Usage:
    python3 scripts/lunarfire_architecture_v07.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.architecture_comparison_v07 import run_architecture_v07_comparison
from helionis.architecture_v07_reporting import (
    plot_architecture_v07,
    write_architecture_v07_csv,
    write_architecture_v07_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire v0.7 same-assumption architecture outputs."
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
    if not math.isfinite(args.target_mw) or args.target_mw <= 0:
        parser.error("--target-mw must be positive and finite")

    output_dir = Path(args.output_dir)
    results = run_architecture_v07_comparison(
        target_screening_net_mw=args.target_mw,
    )
    if not results:
        parser.error("no feasible v0.7 architecture candidates found")

    csv_path = output_dir / "lunarfire_architecture_v0_7.csv"
    markdown_path = output_dir / "lunarfire_architecture_v0_7.md"
    plot_path = output_dir / "lunarfire_architecture_v0_7.png"

    write_architecture_v07_csv(results, csv_path)
    write_architecture_v07_markdown(results, markdown_path)
    if not args.no_plot:
        plot_architecture_v07(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
