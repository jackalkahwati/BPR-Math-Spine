#!/usr/bin/env python3
"""Generate the LunarFire v0.9 BPR-coupled mirror/nozzle screen.

Usage:
    python3 scripts/lunarfire_bpr_coupled_v09.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.bpr_coupled_v09 import run_bpr_coupled_v09_sweep
from helionis.bpr_coupled_v09_reporting import (
    plot_bpr_coupled_v09,
    write_bpr_coupled_v09_csv,
    write_bpr_coupled_v09_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire v0.9 BPR-coupled outputs."
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
        "--limit",
        type=int,
        default=12,
        help="Number of top-ranked rows to write.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.target_mw) or args.target_mw <= 0:
        parser.error("--target-mw must be positive and finite")
    if args.limit <= 0:
        parser.error("--limit must be positive")

    output_dir = Path(args.output_dir)
    results = run_bpr_coupled_v09_sweep(
        target_screening_net_mw=args.target_mw,
        limit=args.limit,
    )
    if not results:
        parser.error("no v0.9 BPR-coupled rows found")

    csv_path = output_dir / "lunarfire_bpr_coupled_v0_9.csv"
    markdown_path = output_dir / "lunarfire_bpr_coupled_v0_9.md"
    plot_path = output_dir / "lunarfire_bpr_coupled_v0_9.png"

    write_bpr_coupled_v09_csv(results, csv_path)
    write_bpr_coupled_v09_markdown(results, markdown_path)
    if not args.no_plot:
        plot_bpr_coupled_v09(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
