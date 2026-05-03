#!/usr/bin/env python3
"""Generate the LunarFire 10 MW FRC reference-design sweep.

Usage:
    python3 scripts/lunarfire_reference_design.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.reference_design import solve_reference_design
from helionis.reference_reporting import (
    plot_reference_candidates,
    write_reference_csv,
    write_reference_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire 10 MW FRC reference-design outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="data/helionis",
        help="Directory for CSV, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Number of top feasible candidates to report.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    output_dir = Path(args.output_dir)
    results = solve_reference_design(limit=args.limit)

    csv_path = output_dir / "lunarfire_reference_design_10mw.csv"
    markdown_path = output_dir / "lunarfire_reference_design_10mw.md"
    plot_path = output_dir / "lunarfire_reference_design_10mw.png"

    write_reference_csv(results, csv_path)
    write_reference_markdown(results, markdown_path)
    if not args.no_plot:
        plot_reference_candidates(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
