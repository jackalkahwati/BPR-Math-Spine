#!/usr/bin/env python3
"""Generate the LunarFire v0.8 mirror/nozzle margin recovery sweep.

Usage:
    python3 scripts/lunarfire_margin_recovery_v08.py --output-dir data/helionis
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helionis.margin_recovery_v08 import run_margin_recovery_v08_sweep
from helionis.margin_recovery_v08_reporting import (
    plot_margin_recovery_v08,
    write_margin_recovery_v08_csv,
    write_margin_recovery_v08_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LunarFire v0.8 margin recovery outputs."
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
        default=24,
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
    results = run_margin_recovery_v08_sweep(
        target_screening_net_mw=args.target_mw,
        limit=args.limit,
    )
    if not results:
        parser.error("no v0.8 margin recovery rows found")

    csv_path = output_dir / "lunarfire_margin_recovery_v0_8.csv"
    markdown_path = output_dir / "lunarfire_margin_recovery_v0_8.md"
    plot_path = output_dir / "lunarfire_margin_recovery_v0_8.png"

    write_margin_recovery_v08_csv(results, csv_path)
    write_margin_recovery_v08_markdown(results, markdown_path)
    if not args.no_plot:
        plot_margin_recovery_v08(results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if not args.no_plot:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
