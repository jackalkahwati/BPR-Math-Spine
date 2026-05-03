#!/usr/bin/env python3
"""Generate the LunarFire v0.2 engineering net-power budget.

Usage:
    python3 scripts/lunarfire_engineering_net.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.engineering_net import run_engineering_net_budget
from helionis.engineering_reporting import (
    plot_engineering_net,
    write_engineering_csv,
    write_engineering_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire engineering net-power outputs."
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
        help="Number of top reference candidates to evaluate.",
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
    results = run_engineering_net_budget(limit=args.limit)

    csv_path = output_dir / "lunarfire_engineering_net_v0_2.csv"
    markdown_path = output_dir / "lunarfire_engineering_net_v0_2.md"
    plot_path = output_dir / "lunarfire_engineering_net_v0_2.png"

    write_engineering_csv(results, csv_path)
    write_engineering_markdown(results, markdown_path)
    if not args.no_plot:
        plot_engineering_net(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
